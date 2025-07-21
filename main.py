import argparse
from dataclasses import asdict

from foc import *
from ouch import *

__version__ = "0.1.0"


class _help_formatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=30)


def main():
    parser = argparse.ArgumentParser(
        prog="nul",
        formatter_class=_help_formatter,
        add_help=False,
    )
    subparser = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="",
    )
    options = parser.add_argument_group("options")

    options.add_argument(
        "--conf",
        action="store_true",
        help="show the default model configuration",
    )
    options.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this message",
    )
    options.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="show version information",
    )

    # new
    new_ = subparser.add_parser(
        "new",
        help="create a model from a conf file",
        formatter_class=_help_formatter,
    )
    new_.add_argument(
        "-f",
        "--file",
        type=str,
        metavar="FILE",
        help="name of configuration file for a model",
    )
    new_.add_argument(
        "-c",
        "--clone",
        type=str,
        metavar="NAME",
        help="create a model by cloning an existing one",
    )
    new_.add_argument(
        "model",
        type=str,
        metavar="MODEL",
        help="model name",
    )
    new_.set_defaults(func=new)

    # ls
    ls_ = subparser.add_parser(
        "ls",
        help="list models",
        formatter_class=_help_formatter,
    )
    ls_.add_argument(
        "--all",
        action="store_true",
        help="include checkpoint models",
    )
    ls_.set_defaults(func=ls)

    # rm
    rm_ = subparser.add_parser(
        "rm",
        help="remove models",
        formatter_class=_help_formatter,
    )
    rm_.add_argument(
        "models",
        type=str,
        nargs="*",
        metavar="MODEL",
        help="model name",
    )
    rm_.set_defaults(func=rm)

    # show
    show_ = subparser.add_parser(
        "show",
        help="show information for a model",
        formatter_class=_help_formatter,
    )
    show_.add_argument(
        "--all",
        action="store_true",
        help="display the model architecture",
    )
    show_.add_argument(
        "model",
        type=str,
        metavar="MODEL",
        help="model name",
    )
    show_.set_defaults(func=show)

    # train
    train_ = subparser.add_parser(
        "train",
        help="train a model",
        formatter_class=_help_formatter,
    )
    train_.add_argument(
        "model",
        type=str,
        metavar="MODEL",
        help="model name",
    )
    train_.add_argument(
        "-f",
        "--file",
        type=str,
        metavar="FILE",
        help="name of configuration file for training",
    )
    train_.set_defaults(func=train)

    # ------------------------------
    args = parser.parse_args()
    if args.version:
        die(f"NUL version is {__version__}")
    elif args.conf:
        dump_def_conf()
    elif hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def new(args):
    """Create a model from a conf file"""
    from nul import dumper, nul, size_model, which_model

    if args.clone:
        path = which_model(clone)
        o = nul.load(clone)
        o.save(args.model)
        print(f"created '{args.model}'")
    else:
        o = nul.new(args.model, conf=args.file)
        o.info(True)
        prompt(
            "\nAre you sure to create a model like this?",
            ok=lazy(
                cf_(
                    lambda x: dumper(
                        dict(
                            model=o.name,
                            path=x,
                            size=size_model(o.name),
                        )
                    ),
                    o.save,
                )
            ),
        )


def ls(args):
    """List models"""
    from nul import list_models

    list_models(all=args.all)


def rm(args):
    """Remove a model"""
    from nul import which_model

    if not args.models:
        die("Error, model name not provided.")

    for model in args.models:
        path = which_model(model)
        if path:
            shell(f"rm -rf {path} {path}.ckpt {path}.train 2>/dev/null")
            print(f"deleted '{model}'")


def show(args):
    """Show information for a model"""
    from nul import nul

    if args.model:
        o = nul.load(args.model)
        o.info(args.all)


def train(args):
    """Train a model"""
    from nul import dumper, nul

    o = nul.load(args.model, conf=args.file)
    print()
    o.info(True)
    t = dmap(
        tset=o.conf.tset,
        vset=o.conf.vset,
        size_batch=o.conf.size_batch,
        lr=o.conf.lr,
        lr_min=o.conf.lr_min,
        warmup=o.conf.warmup,
        optim=o.conf.optim,
        epochs=o.conf.epochs,
        steps=o.conf.steps,
        it=(
            f"{o.it}  "
            f"({100 * o.it / (o.conf.epochs * o.conf.steps):.2f}"
            "% complete)"
        ),
    )
    dumper(t)
    prompt(
        "\nWant to use the above trainer to train the model?",
        fail=lazy(die, "Canceled."),
    )
    o.self_supervised()


def dump_def_conf():
    from nul import nulconf

    conf = asdict(nulconf())
    for k, v in conf.items():
        print(f"{k:<16}  {repr(v)}")


if __name__ == "__main__":
    main()
