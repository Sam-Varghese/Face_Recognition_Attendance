# Importing necessary modules

from rich import print
import tensorflow as tf
import os
import shutil
import cv2 as cv
import random
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import mysql.connector
import json
import datetime

json_data = json.loads(open("./connection_info.json").read())

os.system('cls||clear')

#* Establishing MySQL connection
con = mysql.connector.connect(
    host = json_data["mysql_host"],
    user = json_data["mysql_user"],
    password = json_data["mysql_password"]
)

if con.is_connected():

    print("[green]Connection to database successfully established[/green]")

cursor = con.cursor()
master_database = json_data["master_database"]
cursor.execute("CREATE DATABASE IF NOT EXISTS {};".format(master_database))
cursor.execute("USE {};".format(master_database))

#* Defining global variables
# Directory where the images of faces needs to be stored
facial_database_directory = "facial_database"

#* Checking presence of necessary directories
try:
    os.mkdir(facial_database_directory)

except Exception as e:
    pass

def get_current_time_stamp():

    """
    Return the current date and time in a formatted way.
    """

    time_stamp = datetime.datetime.now()
    formatted_time_stamp = time_stamp.strftime("%d/%m/%Y %H:%M:%S")

    return formatted_time_stamp

def create_role(role_id: str, role_name: str, role_desc: str):

    """
    Create a new role (Ex: Class 6 student, etc)

    Args:

    `role_id`: Unique ID for that role
    `role_name`: Name of the role
    `role_desc`: Optional desc for the role
    """

    cursor.execute("""CREATE TABLE IF NOT EXISTS ROLES (
        ROLE_ID VARCHAR(255) PRIMARY KEY, 
        ROLE_NAME VARCHAR(255) NOT NULL, 
        ROLE_DESC VARCHAR(255) DEFAULT NULL,
        TIME_STAMP VARCHAR(255)
        );""")

    cursor.execute("INSERT INTO ROLES VALUES ('{}', '{}', '{}', '{}');".format(role_id, role_name, role_desc, get_current_time_stamp()))
    con.commit()

    return True

def create_user(user_name: str, role_ids: list, user_id: str, mail_ids: list, contact_numbers: list, address: str):

    # Table for user

    cursor.execute("""CREATE TABLE IF NOT EXISTS USERS (
        USER_NAME VARCHAR(255) NOT NULL, 
        USER_ID VARCHAR(255) PRIMARY KEY, 
        ADDRESS VARCHAR (255) DEFAULT NULL, 
        TIME_STAMP VARCHAR(255) NOT NULL,
        FOREIGN KEY (ROLE_ID) REFERENCES ROLES(ROLE_ID)
        );""")

    # Table for mail ids

    cursor.execute("""CREATE TABLE IF NOT EXISTS MAIL_IDS (
        USER_ID VARCHAR(256), 
        MAIL_ID VARCHAR(256) UNIQUE NOT NULL, 
        FOREIGN KEY (USER_ID) REFERENCES USERS(USER_ID), 
        PRIMARY KEY (USER_ID, MAIL_ID)
        );""")

    # Table for contact numbers

    cursor.execute("""CREATE TABLE IF NOT EXISTS CONTACT_NUMBERS (
        USER_ID VARCHAR(256), 
        CONTACT_NUMBER VARCHAR(20) NOT NULL, 
        FOREIGN KEY (USER_ID) REFERENCES USERS (USER_ID), 
        PRIMARY KEY (USER_ID, CONTACT_NUMBER)
        );""")

    # Table for Roles

    cursor.execute("""CREATE TABLE IF NOT EXISTS USER_ROLES(
        USER_ID VARCHAR(256),
        ROLE_ID VARCHAR(256) NOT NULL,
        FOREIGN KEY USER_ID REFERENCES USERS(USER_ID),
        PRIMARY KEY (USER_ID, ROLE_ID)
    )""")

    # Inserting values

    # Users table
    cursor.execute("INSERT INTO USER VALUES ('{}', '{}', '{}', '{}');".format(user_name, user_id, address, get_current_time_stamp()))
    con.commit()

    # Mail IDs table
    for mail_id in mail_ids:

        cursor.execute("INSERT INTO MAIL_IDS VALUES ('{}', '{}');".format(user_id, mail_id))

    con.commit()

    # Phone numbers table
    for contact_number in contact_numbers:

        cursor.execute("INSERT INTO CONTACT_NUMBERS VALUES ('{}', '{}');".format(user_id, contact_number))

    con.commit()

    # Roles table
    for role_id in role_ids:

        cursor.execute("INSERT INTO USER_ROLES VALUES ('{}', '{}');".format(user_id, role_id))

    con.commit()

def record_face(person_name: str, person_id: str, count: int = 100, sleep_secs: int = 1, re_register = False):

    """
    Function to take images of the faces, and store them in the facial_database_directory.

    ---

    Args:

    ---

    `person_name`: Name of person whose face is being registered

    `count`: Number of images that should be taken, of the person in order to make the mode recognize them

    `sleep_secs`: Number of seconds computer should sleep for after every image taken, so that student can change expression, and include variations in the image inputs

    `re_register`: If the student was already registered, but we need to delete their previous images, and take new ones, then set this parameter True
    """

    image_count = 0
    capture = cv.VideoCapture(0)

    try:
        os.mkdir(os.path.join)
    except Exception as e:
        pass

    person_name = person_name.replace(" ", "-")
    if re_register:

        try:
            shutil.rmtree(os.path.join(facial_database_directory, person_name))
        except Exception:
            pass

    os.mkdir(os.path.join(facial_database_directory, person_name))
    
    while image_count != count:

        isTrue, frame = capture.read()
        print("Saving as ", ".\{}\{}\{}_{}.png".format(facial_database_directory, person_name, image_count, person_name))

        cv.imwrite(".\{}\{}\{}_{}.png".format(facial_database_directory, person_name, image_count, person_name), frame)

        cv.imshow('Video input', frame)

        time.sleep(sleep_secs)

        image_count += 1

        if (cv.waitKey(20) & 0xFF == ord('d')):
            break

    capture.release()
    cv.destroyAllWindows()

    # Updating the recorded face table of MySQL

    cursor.execute("CREATE TABLE IF NOT EXISTS RECORDED_FACES(NAME VARCHAR(255), ID VARCHAR(255), DATE_RECORDED VARCHAR (255));")
    cursor.execute("INSERT INTO RECORDED_FACES VALUES('{}', '{}', '{}')".format(person_name, person_id, get_current_time_stamp()))
    con.commit()

    return True

def face_recognition_model():

    """
    This function makes the machine learning model, and trains it on the facial_database_directory. 80% of the dataset gets used for training, while the rest 20% for validation.

    Note: Currently the model does not support incremental learning. Any efforts to introduce this feature will be highly appreciated.
    """

    batch_size = 32
    img_height = 100
    img_width = 100
    facial_database_member_names = os.listdir(facial_database_directory)
    num_classes = len(facial_database_member_names)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        facial_database_directory,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        facial_database_directory,
        validation_split = 0.2,
        subset = "validation",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    model = tf.keras.Sequential([
        # Data augmentation for better recognition

        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),

        tf.keras.layers.Rescaling(1./255),
        
        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dense(num_classes, activation = "softmax")
    ])

    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_ds, validation_data = val_ds, epochs = 20)

    return model

model = face_recognition_model()