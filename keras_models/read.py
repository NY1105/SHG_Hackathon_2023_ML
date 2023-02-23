import glob
import configuration as config
def read(attribute=None):
	files = config.dims
	if attribute:
		files = [attribute]
	for file in files:
		with open("./res/"+file+".res") as f:
			print(f.read())

if __name__ == "__main__":
	read()