import os

os.environ['PATH_ROOT_DIR'] = '../../data/eeg-motor-movementimagery-dataset-1.0.0/files'
os.environ['PATH_RECORDS'] = '../../data/eeg-motor-movementimagery-dataset-1.0.0/files/RECORDS'
os.environ['FILES_PER_SUBJ'] = "3"
os.environ['NUM_SUBJS'] = "10"
os.environ['EPOCH_MIN'] = "0"
os.environ['EPOCH_MAX'] = "2.0"
os.environ["SAMPLE_RATE"]='160'
os.environ["BATCH_SIZE"]='4'
