# Importing necessary modules

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from rich import print
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from rich.table import Table
import numpy as np
import tensorflow as tf
import shutil
import cv2 as cv
import time
from tensorflow.keras.optimizers import Adam
import mysql.connector
import json
import datetime
import uuid
import pyttsx3

engine = pyttsx3.init()

print("[green]Imported modules[/green]")

try:
    json_data = json.loads(open("./connection_info.json").read())
except Exception as e:
    print("[red]connection_info.json not made[/red] Kindly make a new file namely connection_info.json which will has password to you MySQL database along with other necessary information.\nFormat:")
    print({
    "mysql_password": "your_password",
    "mysql_host": "localhost",
    "mysql_user": "root",
    "master_database": "FRAS"
})
    input("Press enter once you make that file")
    json_data = json.loads(open("./connection_info.json").read())

os.system('cls||clear')

#* Establishing MySQL connection
con = mysql.connector.connect(
    host = json_data["mysql_host"],
    user = json_data["mysql_user"],
    password = json_data["mysql_password"],
    auth_plugin='mysql_native_password'
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
ml_model_directory = "ML_model"

#* Checking presence of necessary directories
try:
    os.mkdir(facial_database_directory)
except Exception as e:
    pass
try:
    os.mkdir(ml_model_directory)
except Exception as e:
    pass

def speak_text(text: str):

    engine.say(text)
    engine.runAndWait()

def get_unique_id():
    """
    This function returns a unique ID. It's guaranteed that this ID will never get returned again by this function, in future.
    """
    return str(uuid.uuid4())

def get_current_time_stamp():

    """
    Return the current date and time in a formatted way.
    """

    time_stamp = datetime.datetime.now()
    formatted_time_stamp = time_stamp.strftime("%d/%m/%Y %H:%M:%S")

    return formatted_time_stamp

def get_dd_mm_yyyy():
    """
    Returns time stamp in the format of dd/mm/yyyy
    """

    time_stamp = datetime.datetime.now()
    formatted_time_stamp = time_stamp.strftime("%d/%m/%Y")

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
        MODEL_INDEX INT DEFAULT NULL
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
        FOREIGN KEY (USER_ID) REFERENCES USERS(USER_ID),
        PRIMARY KEY (USER_ID, ROLE_ID)
    )""")

    # Inserting values

    # Users table
    print("Executing query: ")
    print("INSERT INTO USERS VALUES ('{}', '{}', '{}', '{}', NULL);".format(user_name, user_id, address, get_current_time_stamp()))
    cursor.execute("INSERT INTO USERS VALUES ('{}', '{}', '{}', '{}', NULL);".format(user_name, user_id, address, get_current_time_stamp()))
    con.commit()

    # Mail IDs table
    for mail_id in mail_ids:

        if (mail_id.lower() not in ["null", ""]):
            cursor.execute("INSERT INTO MAIL_IDS VALUES ('{}', '{}');".format(user_id, mail_id))

    con.commit()

    # Phone numbers table
    for contact_number in contact_numbers:

        if(contact_number.lower() not in ["null", ""]):
            cursor.execute("INSERT INTO CONTACT_NUMBERS VALUES ('{}', '{}');".format(user_id, contact_number))

    con.commit()

    # Roles table
    for role_id in role_ids:

        if(role_id.lower() not in ["null", ""]):
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
        os.mkdir(facial_database_directory)
    except Exception as e:
        pass

    person_name = person_name.replace(" ", "-")
    print("Working for person ", person_name, person_id)

    folder_name = [i for i in os.listdir(facial_database_directory) if person_id in i]
    print(folder_name)
    if(len(folder_name) != 0):
        try:
            shutil.rmtree(os.path.join(facial_database_directory, folder_name[0]))
        except Exception:
            pass
    folder_name = str(len(os.listdir(facial_database_directory))) +"_"+ person_id
    
    os.mkdir(os.path.join(facial_database_directory, folder_name))
    
    while image_count != count:

        isTrue, frame = capture.read()
        print("Saving as ", ".\{}\{}\{}_{}.png".format(facial_database_directory, folder_name, image_count, person_name))

        cv.imwrite(".\{}\{}\{}_{}.png".format(facial_database_directory, folder_name, image_count, person_name), frame)

        cv.imshow('Video input', frame)

        time.sleep(sleep_secs)

        image_count += 1

        if (cv.waitKey(20) & 0xFF == ord('d')):
            break

    capture.release()
    cv.destroyAllWindows()

    return True
def face_recognition_model():

    """
    This function makes the machine learning model, and trains it on the facial_database_directory. 80% of the dataset gets used for training, while the rest 20% for validation.

    Note: Currently the model does not support incremental learning. Any efforts to introduce this feature will be highly appreciated.
    """

    batch_size = 32
    img_height = 200
    img_width = 200
    cursor.execute("SELECT USER_ID FROM USERS ORDER BY MODEL_INDEX;")
    facial_database_member_id = [id[0] for id in cursor.fetchall()]
    num_classes = len(facial_database_member_id)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        facial_database_directory,
        validation_split = 0.2,
        subset = "training",
        color_mode = "grayscale",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        facial_database_directory,
        validation_split = 0.2,
        subset = "validation",
        color_mode = "grayscale",
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    model = tf.keras.Sequential([
        # Data augmentation for better recognition

        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        # tf.image.central_crop(0.5),

        tf.keras.layers.Rescaling(1./255),
        
        tf.keras.layers.Conv2D(32, 3, activation = "relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation = "relu"),
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
    model.save(os.path.join(ml_model_directory, "model.h5"))

    print("[green]Model saved successfully[/green]")

    # Updating the Users table MODEL_INDEX attribute to store the order in which predictions are made by ML model.

    counter = 0
    for user_id in facial_database_member_id:

        cursor.execute("UPDATE USERS SET MODEL_INDEX = {} WHERE USER_ID = '{}';".format(counter, user_id))

        counter += 1

    con.commit()

    return True

def predict_frame(model, frame):

    """
    Function to predict the class of frame captured by open-cv.
    
    ---
    
    Args:

    `model`: The trained machine learning model
    `frame`: Frame captured by open-cv

    ---

    Return:

    `Status`: True/ False depending on whether anybody was identified with a certain level of accuracy.
    `User ID`: If the status is True, then function will also return the User ID of the identified user in the frame. Else it'll return None
    """
    successive_feature_maps = model.predict(frame)

    # Get ID's of users in order of successive_feature_maps
    cursor.execute("SELECT USER_ID FROM USERS WHERE MODEL_INDEX IS NOT NULL ORDER BY MODEL_INDEX ASC;")
    try:
        predicted_user_ids = [name[0] for name in cursor.fetchall()]
    except Exception as e:
        print("[red]Fetching Predicted User ID Failed[/red]: [yellow]There might not be any users who have registered their face. Hence kindly make them register face to start FRAS.[/yellow]")
        return [False, False]

    max_predicted_probability = max(successive_feature_maps[0])

    # Confirm a frame to be of a user only if predicted probability comes out to be > 60%
    print("Max probability is: ",max_predicted_probability, successive_feature_maps, predicted_user_ids)
    if(max_predicted_probability <= 0.60):
        return [False, None]

    else:
        return [True, predicted_user_ids[list(successive_feature_maps[0]).index(max_predicted_probability)]]

def identify_image():

    pass

def recognize_face():

    capture = cv.VideoCapture(0)

    cursor.execute("""CREATE TABLE IF NOT EXISTS ATTENDANCE_RECORDS (
        USER_ID VARCHAR(255),
        TIME_STAMP VARCHAR(255),
        ATTENDANCE VARCHAR(255),
        PRIMARY KEY (USER_ID, TIME_STAMP),
        FOREIGN KEY (USER_ID) REFERENCES USERS(USER_ID)
    );""")
    
    try:
        model = tf.keras.models.load_model(os.path.join(ml_model_directory, "model.h5"))
    except Exception as e:
        print("[yellow]Unable to load ML model[/yellow]\nTrying to create a new model...")
        face_recognition_model()
        model = tf.keras.models.load_model(os.path.join(ml_model_directory, "model.h5"))

    while True:

        isTrue, frame = capture.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.resize(frame, (200, 200))
        frame = np.array(frame)
        frame = frame.reshape((1, )+ frame.shape)

        prediction = predict_frame(model, frame)
        
        time.sleep(1)

        if(prediction[0]):
            cursor.execute("SELECT USER_NAME FROM USERS WHERE USER_ID = '{}';".format(prediction[1]))
            identified_person = cursor.fetchall()[0][0].replace("-", " ").title()
            speak_text("Identified {}".format(identified_person))
            print("[green]Identified {}[/green]".format(identified_person))
            cursor.execute("SELECT USER_ID FROM ATTENDANCE_RECORDS WHERE TIME_STAMP LIKE '{}%' AND USER_ID = '{}';".format(get_dd_mm_yyyy(), prediction[1]))
            
            # If attendance of the identified has already been marked anytime in that day, then no need to mark it again
            if(len(cursor.fetchall()) != 0):

                print("[yellow]Presence already marked[/yellow] database update cancelled.")

            else:
            
                cursor.execute("""INSERT INTO ATTENDANCE_RECORDS VALUES (
                '{}',
                '{}',
                'Present'
            );""".format(prediction[1], get_current_time_stamp()))
                con.commit()
                time.sleep(1) # For the user to move out of the camera cause he has been marked present

        if (cv.waitKey(20) & 0xFF == ord('d')):
            print("[yellow]Aborting camera recording[/yellow]")
            break

def get_roles():

    """
    Returns name of roles along with their ID.

    ---

    Return:

    ---

    [(role1, role1_id), (role2, role2_id),.....]
    """

    cursor.execute("SELECT ROLE_NAME, ROLE_ID FROM ROLES;")
    roles = cursor.fetchall()

    return roles

def display_table(table_title: str, columns_list: list, columnwise_values: list):
    """
    Function to display data in table structure.

    ---

    Args:

    ---

    `table_title`: Name of the table
    `columns_list`: List of attributes that should be included in the table
    `columnwise_values`: List of rows. [[row1 values], [row2 values], ...]
    """
    from rich.console import Console
    from rich.table import Table

    table = Table(title = table_title)
    colors = ["cyan", "blue", "purple4", "magenta"]
    import random
    for column_name in columns_list:
        table.add_column(column_name, justify = "center", style=random.choice(colors))

    for i in columnwise_values:
        table.add_row(*i)

    console = Console()
    console.print(table)

    return True

class menu_item_functions:

    """
    Functions to be executed after options get selected from menu bars.
    Naming convention: `menu_{menu_depth}_{menu_option_number}_{item_option_number}`
    """
    try:
        role_names = get_roles()
    except Exception as e:
        role_names = []

    def menu_1_0_1(self):
        """
        Takes necessary inputs, and creates a new role.
        """

        role_name = input("Name of the role please: ")
        role_desc = input("Description of {} role: ".format(role_name))
        role_id = get_unique_id()

        while (role_name in self.role_names):

            print("[yellow]Warning: [/yellow]This role name already exists")
            rename_role = input("Would you like to rename the role? [y/n] ")

            if(rename_role == "y"):

                role_name = input("Name of the role please: ")

            else:

                print("Not an issue, but please remember that the ID provided to this role is {}".format(role_id))


        create_status = create_role(role_id, role_name, role_desc)

        if(create_status):

            print("[green]Role created successfully.[/green]")

    def menu_1_0_0(self):
        """
        Creates a new user.
        """
        self.role_names = get_roles() # Updating roles info
        if(self.role_names == []):

            print("[red]Roles Unavailable[/red]: There are no roles yet for any user, so kindly create a role first.")

            return False

        user_name = input("Enter the name of user: ").title()
        self.role_names = get_roles() # Updating role names
        print("Kindly select the role for {} from the following list".format(user_name))

        counter = 1

        for role in self.role_names:

            print("{}) {} (ID: {})".format(counter, role[0], role[1]))
            counter += 1

        roles_input = input("Enter the indexes of the roles (separated by commas): ")
        roles_input = roles_input.replace(" ", "")
        roles_input = roles_input.split(",")
        roles_input = [int(role)-1 for role in roles_input]
        roles_input = [self.role_names[i][1] for i in roles_input] # This is the list of role ID's that this user should get

        email_ids = []
        email_id = "Dummy"

        while (email_id != ""):

            email_id = input("Enter the mail id (Just press enter if you don't wanna enter anymore ID's): ")
            address = "NULL" if email_id == "" else email_id

            email_ids.append(email_id)

        contact_nums = []
        contact_num = "Dummy"

        while (contact_num != ""):

            contact_num = input("Enter the contact number (Just press enter if you don't wanna enter anymore number's): ")
            address = "NULL" if contact_num == "" else contact_num

            contact_nums.append(contact_num)

        address = input("Enter the address of this user (Or press enter if you're tired of entering so many details :): ")
        address = "NULL" if address == "" else address
        create_user(user_name, roles_input, get_unique_id(), email_ids, contact_nums, address)

    def menu_1_1_0(self):
        """
        Retrieve details of single user.
        """
        counter = 1

        for role in self.role_names:
            print("{}) {}".format(counter, role))

        role_input = int(input("Enter the index of role which the user belongs to: "))

    def menu_1_1_1(self):

        pass

    def menu_1_2_0(self):
        """
        Recording face
        """
        
        cursor.execute("SELECT USER_NAME, USER_ID FROM USERS;")
        name_id = cursor.fetchall()

        counter = 1

        for i in name_id:

            print("{}) {} (ID: {})".format(counter, i[0], i[1]))
            counter += 1

        selected_person = name_id[int(input("Enter the index of the selected person: "))-1]

        cursor.execute("SELECT MODEL_INDEX FROM USERS WHERE USER_ID = '{}';".format(selected_person[0]))
        already_registered = cursor.fetchall()
        print(already_registered)
        already_registered = True if already_registered in ["none", "", []] else False
        print("Already registered: ", already_registered)
        return record_face(selected_person[0], selected_person[1], 100, 1, already_registered)

    def menu_1_2_1(self):
        """
        Start FRAS: Starting face recognition
        """
        recognize_face()

class terminal_menu:

    title = None
    options_list = None
    previous_terminal_menu = None

    def __init__(self, menu_title, options) -> None:
        self.title = menu_title
        self.options_list = options

    def set_previous_terminal_menu(self, menu):
        self.previous_terminal_menu = menu

    def show_previous_terminal_menu(self):
        self.previous_terminal_menu

    def display(self):

        if(self.title != ""):
            print(Panel(Text(self.title, justify = "center")))
        
        counter = 1

        for option in self.options_list:

            print("[blue]{}:[/blue] {}".format(counter, option))
            counter+=1

        print("[cyan]Please enter the task index: [/cyan]", end = " ")
        item_selected = int(input()) - 1 # Because counter started from 1, not 0

        if(item_selected == len(self.options_list) -1):

            self.show_previous_terminal_menu()

        return item_selected

def show_menu_items():

    """
    Function to make a tree like structure of terminal_menu 's.

    Naming convention of terminal_menu 's: Every terminal_menu will be stored with the following name convention: `menu_{depth}_{option_number}`. Please refer to ./menu_tree.png to have an idea of the tree data structure, and option_number which I'm referring to.
    """

    item_fns = menu_item_functions()
    os.system('cls||clear')
    menu_0_0 = terminal_menu(">>>>===FRAS Features===<<<<", ["Create new", "FRAS", "Retrieve details of users", "Retrieve attendance record", "Train model", "Deletion"])

    menu_1_0 = terminal_menu("Creator", ["Create new user", "Create new role", "Back"])
    
    menu_1_1 = terminal_menu("Retriever", ["Retrieve details of single user", "Retrieve details of users with specific roles", "Export role wise data to excel", "Back"])

    menu_1_2 = terminal_menu("FRAS", ["Record new face", "Start FRAS", "Back"])

    menu_1_3 = terminal_menu("[red]Deletion[/red]", ["Delete user", "Delete role", "Back"])

    menu_1_0.previous_terminal_menu = menu_0_0
    menu_1_1.previous_terminal_menu = menu_0_0
    menu_1_2.previous_terminal_menu = menu_0_0
    menu_1_3.previous_terminal_menu = menu_0_0

    menu_0_0_inp = menu_0_0.display()

    match menu_0_0_inp:

        case 0:
            # Create new user/ role
            menu_1_0_inp = menu_1_0.display()

            match menu_1_0_inp:

                case 0:
                    # Create new user
                    item_fns.menu_1_0_0()

                case 1:
                    # Create new role
                    item_fns.menu_1_0_1()

        case 1:
            menu_1_2_inp = menu_1_2.display()

            match menu_1_2_inp:

                case 0:
                    # Recording face
                    item_fns.menu_1_2_0()

                case 1:
                    # Start FRAS
                    item_fns.menu_1_2_1()

        case 2:
            # Retrieve details
            menu_1_1_inp = menu_1_1.display()

            match menu_1_1_inp:

                case 0:
                    # Single user detail retrieval
                    roles_list = get_roles()
                    role_counter = 1

                    for role in roles_list:
                        print("{}) {} (ID: {})".format(role_counter, role[0], role[1]))
                        role_counter += 1

                    selected_role = roles_list[int(input("Select the role which the user belongs to: ")) - 1]

                    cursor.execute("select * from users where user_id in (select user_id from user_roles where role_id = '{}');".format(selected_role[1]))

                    user_counter = 1
                    users_data = cursor.fetchall()
                    for user in users_data:

                        print("{}) {} (ID: {})".format(user_counter, user[0], user[1]))
                        user_counter += 1
                    selected_user = users_data[int(input("Enter the index of user whose details you would like to retrieve: "))-1]
                    display_table(selected_user[0], ["Name", "ID", "Date of Joining"], [[selected_user[0], selected_user[1], selected_user[3]]])
                    input("Press enter if you've viewed this table: ")

                case 1:
                    # Retrieve details of users with a role
                    roles_list = get_roles()
                    roles_counter = 1

                    for role in roles_list:
                        print("{}) {} (ID: {})".format(roles_counter, role[0], role[1]))
                        roles_counter += 1

                    selected_role = roles_list[int(input("Enter the index of the role you selected: ")) - 1]
                    
                    cursor.execute("select * from users where user_id in (select user_id from user_roles where role_id = '{}');".format(selected_role[1]))
                    users_data = cursor.fetchall()
                    data = []
                    for i in users_data:
                        i = list(i)
                        del i[2]
                        del i[-1]
                        data.append(i)
                    users_data = data
                    display_table(selected_role[0], ["Name", "User ID", "Joining Date"], users_data)
                    input("Press enter if you've viewed the data: ")

        case 4:
            face_recognition_model()

        case 5:
            # Deletion
            menu_1_3_inp = menu_1_3.display()

            roles_list = get_roles()

            match menu_1_3:

                case 0:
                    # Delete a user
                    roles_counter = 1
                    for role in roles_list:

                        print("{}) {} (ID: )".format(roles_counter, role[0], role[1]))
                        roles_counter += 1

                    selected_role = roles_list[int(input("Select the role which the user belongs to: ")) - 1]

                    cursor.execute("select * from users where user_id in (select user_id from user_roles where role_id = '{}');".format(selected_role[1]))

                    user_counter = 1
                    users_data = cursor.fetchall()
                    for user in users_data:

                        print("{}) {} (ID: {})".format(user_counter, user[0], user[1]))
                        user_counter += 1
                        
                    selected_user = users_data[int(input("Enter the index of user whose details you would like to retrieve: "))-1]

while True:
    show_menu_items()
