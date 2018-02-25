from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
import csv

def main():
    dirs = [join("./data", d) for d in listdir("./data") \
            if isdir(join("./data", d)) and d != ".git"]
    for n in range(len(dirs)):
        files = [join(dirs[n], f) for f in listdir(dirs[n]) \
                 if isfile(join(dirs[n], f))]
        files_use = [f for f in listdir(dirs[n]) \
                     if isfile(join(dirs[n], f))]
        for m in range(len(files)):
            f = files[m]    
            temp = files_use[m].split('-')
            filename = temp[1]
            person = temp[0]

            try:
                im = Image.open(f)
                width, height = im.size
                pix = im.getdata()
    
                myfile = open("./files/" + filename, 'a+')
                print("open {0}".format(filename))
                with myfile:
                    writer = csv.writer(myfile)
                    mylist = [] 
                    for i in range(width*height):
                        for j in range(len(pix[i])):
                            mylist.append(pix[i][j])
                    mylist.append(person)
                    writer.writerow(mylist) 
            except OSError:
                print("1 mistake")
    
if __name__=='__main__':
    main()
