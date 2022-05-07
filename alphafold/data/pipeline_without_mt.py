# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
import collections
import contextlib
import copy
import dataclasses
import json
import tempfile
import pickle

from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data.mk_temp_features_from_pdb import make_template_features
from alphafold.data import feature_processing
from alphafold.data import msa_pairing
import numpy as np

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features


class DataPipelineWithoutMT:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self):
    """Initializes the data pipeline."""
    pass
    
  def process(self, input_fasta_path: str, template_path:str = None) -> FeatureDict:
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    # make sequence features
    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    # make fake MSA features
    fake_msa = parsers.parse_a3m(f">{input_description}\n{input_sequence}\n")
    msa_features = make_msa_features([fake_msa])

    # make empty template features
    template_features = make_template_features(input_sequence, template_path)

    return {**sequence_features, **msa_features, **template_features}


@dataclasses.dataclass(frozen=True)
class _FastaChain:
  sequence: str
  description: str


def _make_chain_id_map(*,
                       sequences: Sequence[str],
                       descriptions: Sequence[str],
                       ) -> Mapping[str, _FastaChain]:
  """Makes a mapping from PDB-format chain ID to sequence and description."""
  if len(sequences) != len(descriptions):
    raise ValueError('sequences and descriptions must have equal length. '
                     f'Got {len(sequences)} != {len(descriptions)}.')
  if len(sequences) > protein.PDB_MAX_CHAINS:
    raise ValueError('Cannot process more chains than the PDB format supports. '
                     f'Got {len(sequences)} chains.')
  chain_id_map = {}
  for chain_id, sequence, description in zip(
      protein.PDB_CHAIN_IDS, sequences, descriptions):
    chain_id_map[chain_id] = _FastaChain(
        sequence=sequence, description=description)
  return chain_id_map


def _make_template_map(*,
                       sequences: Sequence[str],
                       template_paths: Sequence[str],
                       ) -> Mapping[str, str]:
  """Makes a mapping from chain ID to template."""
  if len(sequences) != len(template_paths):
    raise ValueError('sequences and template paths must have equal length. '
                     f'Got {len(sequences)} != {len(template_paths)}.')
  if len(sequences) > protein.PDB_MAX_CHAINS:
    raise ValueError('Cannot process more chains than the PDB format supports. '
                     f'Got {len(sequences)} chains.')
  chain_template_map = {}
  for chain_id, sequence, template_path in zip(
      protein.PDB_CHAIN_IDS, sequences, template_paths):
    chain_template_map[chain_id] = template_path
  return chain_template_map


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
  with tempfile.NamedTemporaryFile('w', suffix='.fasta') as fasta_file:
    fasta_file.write(fasta_str)
    fasta_file.seek(0)
    yield fasta_file.name


def convert_monomer_features(
    monomer_features: FeatureDict,
    chain_id: str) -> FeatureDict:
  """Reshapes and modifies monomer features for multimer models."""
  converted = {}
  converted['auth_chain_id'] = np.asarray(chain_id, dtype=np.object_)
  unnecessary_leading_dim_feats = {
      'sequence', 'domain_name', 'num_alignments', 'seq_length'}
  for feature_name, feature in monomer_features.items():
    if feature_name in unnecessary_leading_dim_feats:
      # asarray ensures it's a np.ndarray.
      feature = np.asarray(feature[0], dtype=feature.dtype)
    elif feature_name == 'aatype':
      # The multimer model performs the one-hot operation itself.
      feature = np.argmax(feature, axis=-1).astype(np.int32)
    elif feature_name == 'template_aatype':
      feature = np.argmax(feature, axis=-1).astype(np.int32)
      new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
      feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
    elif feature_name == 'template_all_atom_masks':
      feature_name = 'template_all_atom_mask'
    converted[feature_name] = feature
  return converted


def int_id_to_str_id(num: int) -> str:
  """Encodes a number as a string, using reverse spreadsheet style naming.

  Args:
    num: A positive integer.

  Returns:
    A string that encodes the positive integer using reverse spreadsheet style,
    naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
    usual way to encode chain IDs in mmCIF files.
  """
  if num <= 0:
    raise ValueError(f'Only positive integers allowed, got {num}.')

  num = num - 1  # 1-based indexing.
  output = []
  while num >= 0:
    output.append(chr(num % 26 + ord('A')))
    num = num // 26 - 1
  return ''.join(output)


def add_assembly_features(
    all_chain_features: MutableMapping[str, FeatureDict],
    ) -> MutableMapping[str, FeatureDict]:
  """Add features to distinguish between chains.

  Args:
    all_chain_features: A dictionary which maps chain_id to a dictionary of
      features for each chain.

  Returns:
    all_chain_features: A dictionary which maps strings of the form
      `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
      chains from a homodimer would have keys A_1 and A_2. Two chains from a
      heterodimer would have keys A_1 and B_1.
  """
  # Group the chains by sequence
  seq_to_entity_id = {}
  grouped_chains = collections.defaultdict(list)
  for chain_id, chain_features in all_chain_features.items():
    seq = str(chain_features['sequence'])
    if seq not in seq_to_entity_id:
      seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
    grouped_chains[seq_to_entity_id[seq]].append(chain_features)

  new_all_chain_features = {}
  chain_id = 1
  for entity_id, group_chain_features in grouped_chains.items():
    for sym_id, chain_features in enumerate(group_chain_features, start=1):
      new_all_chain_features[
          f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
      seq_length = chain_features['seq_length']
      chain_features['asym_id'] = chain_id * np.ones(seq_length)
      chain_features['sym_id'] = sym_id * np.ones(seq_length)
      chain_features['entity_id'] = entity_id * np.ones(seq_length)
      chain_id += 1

  return new_all_chain_features


def pad_msa(np_example, min_num_seq):
  np_example = dict(np_example)
  num_seq = np_example['msa'].shape[0]
  if num_seq < min_num_seq:
    for feat in ('msa', 'deletion_matrix', 'bert_mask', 'msa_mask'):
      np_example[feat] = np.pad(
          np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
    np_example['cluster_bias_mask'] = np.pad(
        np_example['cluster_bias_mask'], ((0, min_num_seq - num_seq),))
  return np_example


class MultimerDataPipelineWithouMT:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,monomer_data_pipeline: DataPipelineWithoutMT,max_uniprot_hits: int = 50000):
    """Initializes the data pipeline.

    Args:
      monomer_data_pipeline: An instance of pipeline.DataPipeline - that runs
        the data pipeline for the monomer AlphaFold system.
      jackhmmer_binary_path: Location of the jackhmmer binary.
      uniprot_database_path: Location of the unclustered uniprot sequences, that
        will be searched with jackhmmer and used for MSA pairing.
      max_uniprot_hits: The maximum number of hits to return from uniprot.
      use_precomputed_msas: Whether to use pre-existing MSAs; see run_alphafold.
    """
    self._monomer_data_pipeline = monomer_data_pipeline
    self._max_uniprot_hits = max_uniprot_hits

  def _process_single_chain(
      self,
      chain_id: str,
      sequence: str,
      description: str,
      msa_output_dir: str,
      is_homomer_or_monomer: bool,
      template_path:str = None) -> FeatureDict:
    """Runs the monomer pipeline on a single chain."""
    chain_fasta_str = f'>chain_{chain_id}\n{sequence}\n'
    chain_msa_output_dir = os.path.join(msa_output_dir, chain_id)
    if not os.path.exists(chain_msa_output_dir):
      os.makedirs(chain_msa_output_dir)
    with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
      logging.info('Running monomer pipeline on chain %s: %s',
                   chain_id, description)
      chain_features = self._monomer_data_pipeline.process(
          input_fasta_path=chain_fasta_path,
          template_path = template_path)

      # We only construct the pairing features if there are 2 or more unique
      # sequences.
      if not is_homomer_or_monomer:
        all_seq_msa_features = self._all_seq_msa_features(chain_fasta_str)
        chain_features.update(all_seq_msa_features)

    result_file = os.path.join(chain_msa_output_dir, 'chain_features.pkl')
    with open(result_file, 'wb') as f:
      pickle.dump(chain_features, f, protocol=4)
    return chain_features

  def _all_seq_msa_features(self, input_fasta_str):
    """Get MSA features for unclustered uniprot, for pairing."""
    fake_msa = parsers.parse_a3m(input_fasta_str)
    fake_msa = fake_msa.truncate(max_seqs=self._max_uniprot_hits)
    all_seq_features = make_msa_features([fake_msa])
    valid_feats = msa_pairing.MSA_FEATURES + (
        'msa_species_identifiers',
    )
    feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}
    return feats

  def process(self,
              input_fasta_path: str,
              msa_output_dir: str,
              pdb_template_path_list:list=None) -> FeatureDict:
    """Runs alignment tools on the input sequences and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

    template_map_dict = {}
    if not (pdb_template_path_list is None):
      template_map_dict = _make_template_map(sequences=input_seqs,template_paths=pdb_template_path_list)

    chain_id_map = _make_chain_id_map(sequences=input_seqs,
                                      descriptions=input_descs)
    chain_id_map_path = os.path.join(msa_output_dir, 'chain_id_map.json')
    with open(chain_id_map_path, 'w') as f:
      chain_id_map_dict = {chain_id: dataclasses.asdict(fasta_chain)
                           for chain_id, fasta_chain in chain_id_map.items()}
      json.dump(chain_id_map_dict, f, indent=4, sort_keys=True)

    all_chain_features = {}
    sequence_features = {}
    is_homomer_or_monomer = len(set(input_seqs)) == 1
    for chain_id, fasta_chain in chain_id_map.items():
      if fasta_chain.sequence in sequence_features:
        all_chain_features[chain_id] = copy.deepcopy(
            sequence_features[fasta_chain.sequence])
        continue

      if chain_id in template_map_dict.keys():
        template_path = template_map_dict[chain_id]
      else:
        template_path = None

      chain_features = self._process_single_chain(
          chain_id=chain_id,
          sequence=fasta_chain.sequence,
          description=fasta_chain.description,
          msa_output_dir=msa_output_dir,
          is_homomer_or_monomer=is_homomer_or_monomer,
          template_path = template_path)

      chain_features = convert_monomer_features(chain_features,
                                                chain_id=chain_id)
      all_chain_features[chain_id] = chain_features
      sequence_features[fasta_chain.sequence] = chain_features

    all_chain_features = add_assembly_features(all_chain_features)

    np_example = feature_processing.pair_and_merge(
        all_chain_features=all_chain_features)

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pad_msa(np_example, 512)

    return np_example
