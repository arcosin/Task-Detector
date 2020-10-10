
# Imports.
import argparse

from experiment_1_atari.main import main as atariMain
from experiment_2_face_tasks.main import main as faceMain
from experiment_2_face_tasks.Doric.examples.deform_cnn_vae import main as progMain


# Constants.
NAME_STR = ""
DESCRIP_STR = ""


#---------------------------------[module code]---------------------------------


# Needs to run doric face experiment, atari main.py, and task_detector_main.py.
# Also maybe mask gen.
# When it runs, see if you can switch it to just one doric install.



def main(args):
    if args.command == "atari_tasknet":
        atariMain()
    elif args.command == "face_tasknet":
        faceMain()
    elif args.command == "face_prognet":
        progMain(args)
    else:
        print("[Task Detector]:  Unknown error.")






#--------------------------------[module setup]---------------------------------

def buildCLIParser():
    parser = argparse.ArgumentParser(prog = NAME_STR, description = DESCRIP_STR)   # Create module's cli parser.
    #parser.add_argument("", )   # (name or flags... [, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest]).
    subparsers = parser.add_subparsers(dest = "command")
    parserDoric = subparsers.add_parser("face_prognet", help = "Run the Doric face image task example program. Uses a progressive neural net to do reconstruction, denoising, colorizing, and inpainting.")
    parserAtari = subparsers.add_parser("atari_tasknet", help = "Run the task detector for Atari games.")
    parserFace = subparsers.add_parser("face_tasknet", help = "Run the task detector for face image tasks.")
    parserFace.add_argument("--cpu", help="Specify whether the CPU should be used.", type=bool, nargs='?', const=True, default=False)
    parserFace.add_argument("--output", help="Specify where to log the output to.", type=str, default="output")
    parserFace.add_argument("--batch_size", help="Batch size.", type=int, default=100)
    parserFace.add_argument("--epochs", help="Epochs to train.", type=int, default=50)
    parserFace.add_argument("--lr", help="Learning rate.", type=float, default=0.0005)
    return parser



if __name__ == '__main__':
    parser = buildCLIParser()
    args = parser.parse_args()
    main(args)

#===============================================================================
