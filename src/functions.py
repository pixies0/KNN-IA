import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import time
import os

def readData (dataBase , test) : 

    trainData = 0
    testData = 0
    trainLabels = 0
    testLabels = 0

    test = test / 100
    test = 1 - test

    if dataBase == 1 :

        allData = p.read_csv ("./dataBases/iris.data" , names = ["A" , "B" , "C" , "D" , "Flor"])

        trainData , testData = train_test_split (allData , test_size = test, random_state = 40 , shuffle = True)
        
        trainLabels = list (trainData ["Flor"])
        testLabels = list (testData ["Flor"])

        del trainData ["Flor"]
        del testData ["Flor"]

        trainData = normalize (trainData , axis = 0)
        testData = normalize (testData , axis = 0)

    else : 

        allData = p.read_csv ("./dataBases/wine.data" , names = ["Vinho" , "A" , "B" , "C" , "D" , "E" , "F" , "G" , "H" , "I" , "J" , "K" , "L" , "M"])

        trainData , testData = train_test_split (allData , test_size = test , random_state = 40 , shuffle = True)

        trainLabels = list (trainData ["Vinho"])
        testLabels = list (testData ["Vinho"])

        del trainData ["Vinho"]
        del testData ["Vinho"]

        trainData = normalize (trainData , axis = 0)
        testData = normalize (testData , axis = 0)

    return trainData , testData , trainLabels , testLabels

def euclidian (testPoint, trainPoint) :

    dist = 0

    for i in range (len (testPoint)) : 
    
        dist = dist + (testPoint [i] - trainPoint [i]) ** 2
    
    return dist ** 0.5

def findClass (classList) : 

    order = max (classList , key = classList.count)

    return order

def closerClass (testElement , trainData , trainLabels , verify) : 

    minDist = float ("inf")
    minClass = 0
    position = 0

    for i in range (len (trainData)) : 

        dist = euclidian (testElement , trainData [i])

        if dist < minDist and verify [i] : 

            minDist = dist
            minClass = trainLabels [i]
            position = i

    verify [position] = False
    
    return minClass , verify

def main () : 

    flag= True
    while(flag != False):
        dataBase = int(input("Digite qual base de dados voce deseja usar :\n1 - iris\n2 - vinhos\n0 - sair\n"))
        
        if(dataBase == 0):
            flag = False
            return

        k = int (input ("\nQuantos vizinhos você deseja utilizar ?\n"))
        test = int (input ("\nQuantos por cento dos dados você deseja usar como base de conhecimento ?\n"))
        os.system('cls')
        start = time.time ()

        trainData , testData , trainLabels , testLabels = readData (dataBase , test)

        precision = 0

        for x in range (len (testData)) : 

            closerClasses = [0] * k
            verify = [True] * len (trainLabels)

            for y in range (k) : 

                closerClasses [y] , verify = closerClass (testData [x] , trainData , trainLabels, verify)

            if findClass (closerClasses) == testLabels [x] : 

                precision = precision + 1

        end = time.time ()
        timer =  end - start
        rate = (precision / len (testLabels)) * 100
        print("Num Vizinhos: ", k)
        print("Base de conhecimento: ", test, "%")
        print ("Tempo de execucao = " , timer)
        print ("\nAcertos : " , precision , "\nTaxa de acertos: " , rate , "%")
        os.system("PAUSE")
        os.system("cls")

