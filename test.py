#!/ifshome/vgupta/miniconda3/bin/python3.5
import sys, getopt
import numpy as np 
import random 

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print ('Input file is "', inputfile)
   print ('Output file is "', outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])

main(sys.argv[1:])

f_file=open(input_file, 'r')
lines_in_file=f_file.readlines()
for i in range(0, 200):
    random.shuffle(lines_in_file)

f_rand = open(output_file, 'w')
for lines in lines_in_file:
    f_rand.write(lines)

f_rand.close()
