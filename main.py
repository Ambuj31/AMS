import os  
import check_camera
import Capture_Image
import Train_Image
import Recognize


def title_bar():
    os.system('cls')  
    print("Face Recognition System")


def mainMenu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("1. Check Camera")
    print("2. Capture Faces")
    print("3. Train Images")
    print("4. Recognize & Attendance")
    print("5. Auto Mail")
    print("6. Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                checkCamera()
                break
            elif choice == 2:
                CaptureFaces()
                break
            elif choice == 3:
                Trainimages()
                break
            elif choice == 4:
                RecognizeFaces()
                break
            elif choice == 5:
                os.system("py automail.py")
                break
                mainMenu()
            elif choice == 6:
                print("Thank You, Have a Great Day!!!")
                break
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")
    exit


def checkCamera():
    check_camera.camer()
    key = input("Enter any key to return main menu")
    mainMenu()


def CaptureFaces():
    Capture_Image.takeImages()
    key = input("Enter any key to return main menu")
    mainMenu()


def Trainimages():
    Train_Image.TrainImages()
    key = input("Enter any key to return main menu")
    mainMenu()


def RecognizeFaces():
    Recognize.recognize_attendence()
    key = input("Enter any key to return main menu")
    mainMenu()


mainMenu()
