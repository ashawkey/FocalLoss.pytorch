import os
import sys
import glob
import signal
import smtplib
import shutil
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders

from .io import load_json

class DelayedKeyboardInterrupt(object):
    """ Delayed SIGINT 
    reference: https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py/21919644
    with statement: 
        * call __init__, instantiate context manager class with parameters.
        * call __enter__, return of __enter__ is assigned to variable after `as`.
        * run `with` body.
        * call __exit__(exc_type, exc_value, exc_traceback). If return false, then raise error, else omit.
        * call __del__
    signal.signal(sig, handler)
        set handler for sig, return the old handler(signal.SIG_DFL)
    """
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received: 
            self.old_handler(*self.signal_received)


class EmailSender(object):
    def __init__(self, subject="model", config="/home/hawkey/my-python-modules/hawtorch/email.json"):
        args = load_json(config)
        self.subject = subject
        self.username = args["username"]
        self.password = args["password"]
        self.send_from = args["send_from"]
        self.send_to = args["send_to"]
        self.server = args["server"]
        self.port = args["port"]

    
    def send(self, files=[], subject=None, message="report", use_tls=True):
        msg = MIMEMultipart()
        msg['From'] = self.send_from
        msg['To'] = COMMASPACE.join(self.send_to)
        msg['Date'] = formatdate(localtime=True)
        subject = self.subject if subject is None else subject
        msg['Subject'] = subject

        message = f"Report from your model {subject}, attachments are: \n {files}"

        msg.attach(MIMEText(message))

        for path in files:
            part = MIMEBase('application', "octet-stream")
            with open(path, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                            'attachment; filename="{}"'.format(os.path.basename(path)))
            msg.attach(part)

        smtp = smtplib.SMTP(self.server, self.port)
        if use_tls:
            smtp.starttls()
        smtp.login(self.username, self.password)
        smtp.sendmail(self.send_from, self.send_to, msg.as_string())
        smtp.quit()

def fix_random_seed(seed=42, cudnn=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def summary(model, input_size, batch_size=-1, device="cuda", logger=None):
    # redirect to write in file
    if logger is not None:
        print = logger._print

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary

def backup(workspace, patterns=["*.py", "*.json", "*.sh"], recursive=True):
    save_path = os.path.join(workspace, "backup")
    os.makedirs(save_path, exist_ok=True)
    for patt in patterns:
        files = glob.glob(patt, recursive=True)
        for f in files:
            if os.path.exists(f):
                ff = os.path.join(save_path, f.split('/')[-1])
                shutil.copy2(f, ff)
    print(f"[INFO] backed up files.")
