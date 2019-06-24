import sys, argparser

def main(argv):
	x_file = options.xfile
	
	# TODO: use x_file to run a GAN on lesion images

if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='Run Lesion GAN')
	parser.add_argument('--xfile', type=str, nargs='?')
	parser.add_argument('--test', dest-'feature', action='store_true')
	options = parser.parse_args()

	main(options)

