import custom
from custom import criterion
from custom.layers import *
from custom.config import config
from model import MusicTransformer
from data import Data
import utils
from midi_neural_processor.processor import decode_midi, encode_midi

import datetime
import argparse
import os

from tensorboardX import SummaryWriter


parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda:1')
else:
    config.device = torch.device('cpu')


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)
mt.load_state_dict(torch.load(os.path.join(args.model_dir, config.checkpoint_pth), map_location=config.device))
mt.test()

if config.condition_file is not None:
    inputs = np.array([encode_midi(config.condition_file)[:500]])
else:
    inputs = np.array([[24, 28, 31]])
inputs = torch.from_numpy(inputs)
result = mt(inputs, config.length, gen_summary_writer)

for i in result:
    print(i, end=' ')

save_path = gen_log_dir + f'/generated-{os.path.splitext(config.checkpoint_pth)[0]}.mid'

decode_midi(result, file_path=save_path)

gen_summary_writer.close()
