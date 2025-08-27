from .bnb import BnBQuantizer
from .gptq import GPTQQuantizer
from .aqlm import AQLMQuantizer
from .awq import AWQQuantizer

QUANTIZER_REGISTRY = {
    "bnb": BnBQuantizer,
    "gptq": GPTQQuantizer,
    "aqlm": AQLMQuantizer,
    "awq": AWQQuantizer
}
