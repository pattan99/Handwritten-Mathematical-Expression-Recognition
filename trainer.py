import torch

def train(model, data_loader, optimizer, criterion, device, logger, debug = False, verbose_freq = 300):
    model.train()
    epoch_loss = 0
    step_cnt = 0

    for batch in data_loader:
        src = batch['src']
        tgt = batch['tgt']
        img = batch['img']

        src_msk = batch['src_msk']
        tgt_msk = batch['tgt_msk']
        src_msk = src_msk.to(device)
        tgt_msk = tgt_msk.to(device)

        input_lengths = torch.sum(dim = 1, input = torch.exp(src_msk), dtype = torch.int32)
        output_lengths = torch.sum(dim = 1, input = torch.exp(tgt_msk), dtype = torch.int32) - 2

        src, tgt, img = src.to(device), tgt.to(device), img.to(device)

        src = src.unsqueeze(1)
        tgt = tgt.permute(1, 0).long()

        optimizer.zero_grad()

        output = model(src, img)  # (seq_len, batch_size, output_dimm)
        # output = output.permute(1, 2, 0)

        tgt = tgt.permute(1, 0)
        tgt = tgt[:, 1:]

        loss = criterion(output, tgt, input_lengths, output_lengths)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        step_cnt+=1

        if step_cnt % verbose_freq == 0:
            logger.info(f'{step_cnt}-th step, loss: {loss.item()}')

    return epoch_loss / len(data_loader)

import re
import jiwer

_COMMAND_RE = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')
def tokenize_expression(s: str) -> list[str]:
  r"""Transform a Latex math string into a list of tokens.

  Tokens are strings that are meaningful in the context of Latex
  e.g. '1', r'\alpha', r'\frac'.

  Args:
    s: unicode input string (ex: r"\frac{1}{2}")

  Returns:
    tokens: list of tokens as unicode strings.
  """
  tokens = []
  while s:
    if s[0] == '\\':
      tokens.append(_COMMAND_RE.match(s).group(0))
    else:
      tokens.append(s[0])

    s = s[len(tokens[-1]) :]

  return tokens

def compute_cer(truth_and_output: list[tuple[str, str]]):
  """Computes CER given pairs of ground truth and model output."""
  class TokenizeTransform(jiwer.transforms.AbstractTransform):
    def process_string(self, s: str):
      return tokenize_expression(r'{}'.format(s))

    def process_list(self, tokens: list[str]):
      return [self.process_string(token) for token in tokens]

  ground_truth, model_output = zip(*truth_and_output)

  return jiwer.cer(truth=list(ground_truth),
            hypothesis=list(model_output),
            reference_transform=TokenizeTransform(),
            hypothesis_transform=TokenizeTransform(),
      )

from tqdm import tqdm
from utils import VectorizeChar

vectorizer = VectorizeChar(257)
vocab = vectorizer.get_vocabulary()

def decode_label(output):
    return "".join([vocab[j] for j in output])

def evaluate(model, data_loader, device):
    model.eval()
    error_char = 0
    # i = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            src = batch['src']
            tgt = batch['tgt']
            img = batch['img']
    
            src, tgt, img = src.to(device), tgt.to(device), img.to(device)
    
            src = src.unsqueeze(1)

            output = model.predict(src, img)

            label_tgt = decode_label(tgt[0][1:-1])
            label_output = decode_label(output[0])

            try:
                err = compute_cer([(label_output, label_tgt)])
            except:
                err = 1

            # print(label_tgt)
            # print(label_output)
            # print(err)
            # print()
            error_char += err
            # i += 1

            # if i == 3:
            #     break
        return error_char / len(data_loader)