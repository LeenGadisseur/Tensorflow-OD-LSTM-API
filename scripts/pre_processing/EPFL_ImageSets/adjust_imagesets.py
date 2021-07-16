

def new_line10(lines_10):
	new_line = ''
	for l in lines_10:
		new_line += l.rstrip('\n') + ','
	#print(new_line)
	line_10 = new_line[:-1]
	line_10 = line_10 + '\n'
	#print(line_10)
	return line_10

def read_write_file():
	fin = open('test_EPFL_seqs_list.txt','r')
	fout =  open('test_EPFL_seqs_list_10.txt','w')
	lines = fin.readlines()
	lines_10 = []
	for indx, line in enumerate(lines):
		
		if((indx+1)%10 ==0 and indx>=9 ):
			lines_10.append(line)
			print("Index: ", indx)
			#print("lines_10: ", lines_10)
			new_line = new_line10(lines_10)
			fout.write(new_line)

			lines_10 = []

		else:
			lines_10.append(line)
	fin.close()
	fout.close()

if __name__=="__main__":
	read_write_file()
