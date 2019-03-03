correctInfoFile = open("../data/test/bounding_box.txt", "r")
correctInfoLines = correctInfoFile.readlines()
calculatedInfoFile = open("../data/test/localization_inf.txt", "r")
calculatedInfoLines = calculatedInfoFile.readlines() 

count = 0

for imageIndex in range(0,100):
    coordinates1 = correctInfoLines[imageIndex].split(",")
    coordinates2 = calculatedInfoLines[imageIndex].split(",")
    x1 = int(coordinates1[1])
    y1 = int(coordinates1[2])
    x2 = int(coordinates1[3])
    y2 = int(coordinates1[4])

    a1 = float(coordinates2[0])
    b1 = float(coordinates2[1])
    a2 = float(coordinates2[0]) + float(coordinates2[2])
    b2 = float(coordinates2[1]) + float(coordinates2[3])

   
    if( x1 > a1):
        leftCornerX =  x1
    else:
        leftCornerX = a1
    if( x2 > a2):
        rightCornerX =  a2
    else:
        rightCornerX = x2
    if( y1 > b1):
        leftCornerY =  y1
    else:
        leftCornerY = b1
    if( y2 > b2):
        rightCornerY =  b2
    else:
        rightCornerY = y2


    w = rightCornerX - leftCornerX

    if( w < 0):
         w = 0

    h = rightCornerY - leftCornerY

    if( h < 0):
        h = 0

    area = w * h
    
    firstArea = (x2 - x1 ) * ( y2 - y1 )
    secondArea = (a2 - a1 ) * ( b2 - b1 )

    union = firstArea + secondArea - area

    result = 100*(area/union)

    print("--- ", imageIndex, " ---")
    print("percentage: ", result, "%")

    if (result > 50):
        count = count + 1

print("\nNumber of correct BBs: ", count)
print("Localization accuracy: ", count/100)
    
        
    