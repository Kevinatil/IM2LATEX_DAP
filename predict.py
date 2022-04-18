import os
from functools import partial
import argparse
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from build_vocab import Vocab, load_vocab
from utils import collate_fn
from model.model import Im2LatexModel
from model.decoding import LatexProducer
from model.score import score_files

class PredictData(Dataset):
    def __init__(self,dir):
        self.dir=dir
        self.images=os.listdir(self.dir) #all the images in the dir
        self.transform=transforms.ToTensor()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.dir,self.images[index])
        img=Image.open(img_path)
        #label=img_path.split('\\')[-1].split('.')[0]
        return self.transform(img)

def main():

    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument('--model_path', required=True,
                        help='path of the evaluated model')
    parser.add_argument("--data_path", type=str,
                        default="./dataset", help="The dataset's dir")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--result_path", type=str,
                        default="./results/predict.txt", help="The file to store result")
    parser.add_argument("--max_len", type=int,
                        default=64, help="Max step of decoding")

    args = parser.parse_args()

    # load model
    checkpoint = torch.load(os.path.join(args.model_path))
    model_args = checkpoint['args']

    # load vocab
    vocab = load_vocab(args.data_path)
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    # load data loader
    data_loader = DataLoader(
        #Im2LatexDataset(args.data_path, args.split, args.max_len),
        PredictData(os.path.join(args.data_path,'image')),
        batch_size=args.batch_size,
        #collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=4
    )

    # load model
    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    result_file = open(args.result_path, 'w')

    # load producer
    latex_producer = LatexProducer(
        model, vocab, model_args.dec_rnn_h,
        max_len=args.max_len, beam_size=args.beam_size)

    for imgs in tqdm(data_loader):
        results = latex_producer(imgs)
        print('\n'.join(results))

        result_file.write('\n'.join(results))
        result_file.write('\n')

    result_file.close()


if __name__ == "__main__":
    main()
