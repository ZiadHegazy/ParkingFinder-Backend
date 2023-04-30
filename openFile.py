sumTime=0
arr=[]
str=""
with open("D:/bachelor/datasets/yolov8statistics.tmdx","r") as file:
    for line in file:

        if (line.__contains__("TP:")):
            str = str +  line.removesuffix("\n")[3:]
        if (line.__contains__("FP:")):
            str = str + "&" + line.removesuffix("\n")[3:]
        if (line.__contains__("FN:")):
            str = str + "&" + line.removesuffix("\n")[3:]
        if (line.__contains__("precision:")):
            str = str + "&" + line.removesuffix("\n")[10:]
        if (line.__contains__("recall:")):
            str = str + "&" + line.removesuffix("\n")[7:]
        if(line.__contains__("time:")):
            str=str+"&"+line.removesuffix("\n")[5:]
            arr.append(str)
            str=""
print(len(arr))
for i in arr :
    print(i)