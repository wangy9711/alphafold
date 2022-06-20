"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.relax import relax
import numpy as np

import tensorflow as tf
import jax

# Internal import (7716).

logging.set_verbosity(logging.INFO)

single_data_dir = '/home/public/af2_database'
multimer_data_dir = '/home/public/af2_database'
param_dir = '/home/public/af2_database/params_new'


# Required parameter
# Input 
flags.DEFINE_string('input_fasta', None, 'Input fasta file')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')
# Database
flags.DEFINE_string('single_data_dir', single_data_dir, 'Path to af2 database')      
flags.DEFINE_string('multimer_data_dir', multimer_data_dir, 'Path to multimer database') 
flags.DEFINE_string('param_dir', param_dir, 'Path to param database') 
flags.DEFINE_string('cus_pdb_seqref', None, 'Path to a directory that will store the results.')

# Optional parameter
flags.DEFINE_integer('recycle', 3, 'recycle times in model inference.')

flags.DEFINE_boolean('use_gpu_relax', True, 'Use gpu to relax')

# max template date
flags.DEFINE_string('max_template_date', '2030-05-01', 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
# tools
flags.DEFINE_string('jackhmmer_binary_path', 'jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', 'hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', 'hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path', 'hmmsearch',
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path', 'hmmbuild',
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path', 'kalign',
                    'Path to the Kalign executable.')

# random seed
flags.DEFINE_integer('seed', 0, 'seed for python reproducibility.')


FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
NUM_ENSEMBLE = 1

def set_device_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    random_seed: int):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  if not os.path.isfile(features_output_path):
    t_0 = time.time()
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        msa_output_dir=msa_output_dir)
    timings['features'] = time.time() - t_0

    # Write out features as a pickled dictionary.
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
  
  else:
    t_0 = time.time()
    f = open(features_output_path, 'rb')
    feature_dict = pickle.load(f) 
    timings['features'] = time.time() - t_0

  unrelaxed_pdbs = {}
  relaxed_pdbs = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict,
                                             random_seed=model_random_seed)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
        model_name, fasta_name, t_diff)

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)

    plddt = prediction_result['plddt']
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode)

    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

    if amber_relaxer:
      # Relax the prediction.
      t_0 = time.time()
      try:
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
      except:
        logging.warning(f'{fasta_name} in {model_name} relaxed failed!')
        relaxed_pdb_str = unrelaxed_pdbs[model_name]
      timings[f'relax_{model_name}'] = time.time() - t_0

      relaxed_pdbs[model_name] = relaxed_pdb_str

      # Save the relaxed PDB.
      relaxed_output_path = os.path.join(
          output_dir, f'relaxed_{model_name}.pdb')
      with open(relaxed_output_path, 'w') as f:
        f.write(relaxed_pdb_str)

  # Rank by model confidence and write out relaxed PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if amber_relaxer:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(json.dumps(
        {label: ranking_confidences, 'order': ranked_order}, indent=4))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check for duplicate FASTA file names.
  random_seed = FLAGS.seed
  set_device_seed(random_seed)
  fasta_path = FLAGS.input_fasta
  fasta_name = pathlib.Path(fasta_path).stem

  output_dir = FLAGS.output_dir

  single_data_dir = FLAGS.single_data_dir
  multimer_data_dir = FLAGS.multimer_data_dir
  param_dir = FLAGS.param_dir
  
  uniref90_database_path = os.path.join(single_data_dir, 'uniref90', 'uniref90.fasta')
  mgnify_database_path = os.path.join(single_data_dir, 'mgnify', 'mgy_clusters.fa')
  bfd_database_path = os.path.join(single_data_dir, 'bfd','bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')
  uniclust30_database_path = os.path.join(single_data_dir, 'uniclust30', 'uniclust30_2018_08', 'uniclust30_2018_08')

  template_mmcif_dir = os.path.join(multimer_data_dir, 'pdb_mmcif', 'mmcif_files')
  obsolete_pdbs_path = os.path.join(multimer_data_dir, 'pdb_mmcif', 'obsolete.dat')
  if FLAGS.cus_pdb_seqref is None:
    pdb_seqres_database_path = os.path.join(multimer_data_dir, 'pdb_seqres.txt')
  else:
    pdb_seqres_database_path = FLAGS.cus_pdb_seqref
  uniprot_database_path = os.path.join(multimer_data_dir, 'uniprot.fasta')

  template_searcher = hmmsearch.Hmmsearch(
      binary_path=FLAGS.hmmsearch_binary_path,
      hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
      database_path=pdb_seqres_database_path)
  template_featurizer = templates.HmmsearchHitFeaturizer(
      mmcif_dir=template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=obsolete_pdbs_path)
  

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=uniref90_database_path,
      mgnify_database_path=mgnify_database_path,
      bfd_database_path=bfd_database_path,
      uniclust30_database_path=uniclust30_database_path,
      small_bfd_database_path='',
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=False,
      use_precomputed_msas=False)

  data_pipeline = pipeline_multimer.DataPipeline(
      monomer_data_pipeline=monomer_data_pipeline,
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      uniprot_database_path=uniprot_database_path,
      use_precomputed_msas=False)


  model_runners = {}
  model_names = config.MODEL_PRESETS['multimer']
  for model_name in model_names:
    model_config = config.model_config(model_name)
    model_config.model.num_ensemble_eval = NUM_ENSEMBLE
    model_config.model.num_recycle=FLAGS.recycle
    
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=param_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  amber_relaxer = relax.AmberRelaxation(
    max_iterations=RELAX_MAX_ITERATIONS,
    tolerance=RELAX_ENERGY_TOLERANCE,
    stiffness=RELAX_STIFFNESS,
    exclude_residues=RELAX_EXCLUDE_RESIDUES,
    max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
    use_gpu=FLAGS.use_gpu_relax)


  predict_structure(
      fasta_path=fasta_path,
      fasta_name=fasta_name,
      output_dir_base=output_dir,
      data_pipeline=data_pipeline,
      model_runners=model_runners,
      amber_relaxer=amber_relaxer,
      random_seed=random_seed)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'input_fasta',
      'output_dir',
  ])

  app.run(main)
