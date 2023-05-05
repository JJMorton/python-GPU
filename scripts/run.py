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
parser.add_argument("cmap", nargs="+", type=str)
args = parser.parse_args()

if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

data = mandy.sample.area(
	args.real, args.imag, args.width, args.height, args.scale, args.max_iters
)

def display(data):
	shape = data.shape
	buffer = ""
	for im in reversed(range(shape[0])):
		for re in range(shape[1]):
			buffer += f"{data[im, re]:4.0f}"
		buffer += "\n"
	print(buffer)

# Convert to an image
cmap = mandy.colour.build_colour_map(args.cmap, 256)

starttime = timeit.default_timer()
img = mandy.colour.image(data, args.max_iters, cmap)
duration = timeit.default_timer() - starttime
print(f"Calculated in {duration:e}s")

mandy.colour.encode(img).save(os.path.join(OUTPUT_DIR, "mandybrot.png"))
