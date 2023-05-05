import argparse
import os
import timeit

import python_gpu as mandy

OUTPUT_DIR = "output"

parser = argparse.ArgumentParser()
parser.add_argument("real", type=float)
parser.add_argument("imag", type=float)
parser.add_argument("width", type=int)
parser.add_argument("height", type=int)
parser.add_argument("scale", type=float)
parser.add_argument("max_iters", type=int)
args = parser.parse_args()

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

starttime = timeit.default_timer()
data = mandy.sample.area(
	args.real, args.imag, args.width, args.height, args.scale, args.max_iters
)
duration = timeit.default_timer() - starttime
print(f"Calculated in {duration:e}s")

def display(data):
	shape = data.shape
	buffer = ""
	for im in reversed(range(shape[0])):
		for re in range(shape[1]):
			buffer += f"{data[im, re]:4.0f}"
		buffer += "\n"
	print(buffer)

display(data)
