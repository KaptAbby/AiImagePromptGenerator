import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import streamlit as st
import json
import requests
import time
from io import BytesIO
from PIL import Image
import base64
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import IntegrityError

def load_mnist(path, kind='train'):
    import gzip
    
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28, 1)
   
    return images, labels

def load_model():
    
    tf.compat.v1.disable_eager_execution()
    # Load MNIST data
    print("####### Loading my data ########")
    train_images, train_labels = load_mnist(r'C:\Users\Abby\Downloads\images', kind='train')
    test_images, test_labels = load_mnist(r'C:\Users\Abby\Downloads\images', kind='t10k')

    # Normalize images
    train_images = train_images.astype(np.float32)
    train_images = train_images * 2 / 255.0 - 1  # Normalize to [-1, 1]

    # Convert labels to one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels)      
    test_labels = tf.keras.utils.to_categorical(test_labels)

    # Define the generator
    print("####### defining the generator and discriminator ########")
    def generator(z, reuse=None):
        with tf.compat.v1.variable_scope('generator', reuse=reuse):
            hidden1 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)(z)
            hidden2 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)(hidden1)
            output = tf.keras.layers.Dense(units=784, activation=tf.nn.tanh)(hidden2)
            return output

    # Define the discriminator
    def discriminator(X, reuse=None):
        with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
            hidden1 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)(X)
            hidden2 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)(hidden1)
            logits = tf.keras.layers.Dense(units=1)(hidden2)
            output = tf.sigmoid(logits)
            return logits

    # Input placeholders
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
    z = tf.compat.v1.placeholder(tf.float32, shape=[None, 100])

    # Start the GAN
    print("####### Training the GAN model ########")
    fake_x = generator(z)
    D_logits_real = discriminator(x) 
    D_logits_fake = discriminator(fake_x, reuse=True)

    # Discriminator Loss
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real, labels=tf.ones_like(D_logits_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.zeros_like(D_logits_fake))) 
    D_loss = D_loss_real + D_loss_fake

    # Generator Loss
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake, labels=tf.ones_like(D_logits_fake)))

    # Optimizing the Loss
    theta_D = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    theta_G = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    D_optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(D_loss, var_list=theta_D)
    G_optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(G_loss, var_list=theta_G)

    # Training the GAN
    batch_size = 100
    num_epochs = 1000
    num_batches = train_images.shape[0] // batch_size

# Database setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(120), unique=True, nullable=False)
    phone = Column(String(20))
    password = Column(String(255))  # In practice, ensure this is hashed

class ImageRequest(Base):
    __tablename__ = 'imagerequests'

    id = Column(Integer, primary_key=True)  # Primary key for ImageRequest
    user_id = Column(Integer, ForeignKey('users.id'))  # Foreign key referencing 'users' table
    description = Column(String(255))  # Description of the image request

    # Establish a relationship with the User class
    user = relationship('User', back_populates='image_requests')

# Establishing the back_populates attribute in the User class
User.image_requests = relationship('ImageRequest', order_by=ImageRequest.id, back_populates='user')
class GeneratedImages(Base):
    __tablename__ = 'generatedimages'

    id = Column(Integer, primary_key=True)  
    request_id = Column(Integer, ForeignKey('imagerequests.id'))  
    imageUrl = Column(String(255))  

    # Establish a relationship with the User class
    request = relationship('ImageRequest', back_populates='generated_images')
# Establishing the back_populates attribute in the User class
ImageRequest.generated_images = relationship('GeneratedImages', order_by=GeneratedImages.id, back_populates='request')

class Roles(Base):
    __tablename__ = 'roles'

    id = Column(Integer, primary_key=True)  
    user_id = Column(Integer, ForeignKey('users.id'))  
    userRole = Column(String(255))  

    # Establish a relationship with the User class
    role = relationship('User', back_populates='user_role')
# Establishing the back_populates attribute in the User class
User.user_role = relationship('Roles', order_by=Roles.id, back_populates='role')

class Feedback(Base):
    __tablename__ = 'feedback'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    message = Column(String(500))
    # type = Column(String(20))  # 'feedback' or 'complaint'
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship('User', back_populates='feedbacks')

User.feedbacks = relationship('Feedback', order_by=Feedback.id, back_populates='user')

# Create engine and session
DATABASE_URL = "mysql+mysqlconnector://root@localhost/aiimagegenerator"
engine = create_engine(DATABASE_URL)

# Create tables
# Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
def is_admin(user_id):
    session = Session()
    role = session.query(Roles).filter_by(user_id=user_id).first()
    session.close()
    return role and role.userRole == 'admin'

def admin_page():
    st.title("Administrator Dashboard")
    
    if 'user_id' not in st.session_state or not is_admin(st.session_state.user_id):
        st.error("You don't have permission to access this page.")
        return

    st.sidebar.button("Back to Home", key="back_to_home", on_click=lambda: setattr(st.session_state, 'page', 'home'))

    tab1, tab2 = st.tabs(["Registered Users", "Feedback and Complaints"])

    with tab1:
        st.header("Registered Users")
        session = Session()
        users = session.query(User).all()
        
        if users:
            user_data = []
            for user in users:
                role = session.query(Roles).filter_by(user_id=user.id).first()
                user_data.append({
                    "ID": user.id,
                    "Name": f"{user.first_name} {user.last_name}",
                    "Email": user.email,
                    "Phone": user.phone,
                    "Role": role.userRole if role else "N/A"
                })
            st.table(user_data)
        else:
            st.info("No registered users found.")

    with tab2:
        st.header("Feedback and Complaints")
        feedbacks = session.query(Feedback).all()
        
        if feedbacks:
            feedback_data = []
            for feedback in feedbacks:
                feedback_data.append({
                    "ID": feedback.id,
                    "User": f"{feedback.user.first_name} {feedback.user.last_name}",
                    # "Type": feedback.type,
                    "Message": feedback.message,
                    "Date": feedback.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            st.table(feedback_data)
        else:
            st.info("No feedback or complaints found.")

def generate_image(prompt, output_file=None):
    with open("project.json", "r") as file:
        project_config = json.load(file)
    print("Generating image with prompt : " + prompt)
    try: 
        myModelConfig = project_config["image_setup"]["stable_diffusion_url"]
        image_request_settings = project_config["image_settings"]
        image_request_settings["prompt"] = prompt
        
        payload = {**image_request_settings}
        print("----------------------")
        response = requests.post(url=myModelConfig, json=payload)
        print(response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    print("Response : ")
    print( response)
    print(response.json())
    time.sleep(10)
    # st.success("image fetching complete")
    return response.json()
    # return

def registration_page():
    st.title("REGISTER")
    st.subheader("Create a new account")

    with st.form("registration_form"):
        first_name = st.text_input("First Name", placeholder="eg Abby")
        last_name = st.text_input("Last Name", placeholder="eg Kaptern")
        email = st.text_input("Email", placeholder="Username or email")
        phone = st.text_input("Phone Number", placeholder="+2547*******")
        password = st.text_input("Password", type="password", placeholder="Password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Password")

        col1, col2 , col3= st.columns(3)
        with col1:
            cancel = st.form_submit_button("Cancel")
        with col2:
            submit = st.form_submit_button("Register")
        with col3:
            login = st.form_submit_button("Login")

    if submit:
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            # Create a new database session
            session = Session()
            
            try:
                # Create new user
                new_user = User(
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    password=password  # In practice, ensure this is hashed
                )
                session.add(new_user)
                session.commit()
                st.session_state.user_id = new_user.id 
                st.success("Registration successful! click register again to continue")
                st.session_state.page = "home"
            except IntegrityError:
                st.error("A user with this email already exists")
                session.rollback()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                session.rollback()
            finally:
                session.close()
    
    if login:
        st.session_state.page = "login"
        st.success("Click again to proceed")

def login_page():
    # load_model()
    st.title("Login")
    st.subheader("Log in to your account")

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="Username or email")
        password = st.text_input("Password", type="password", placeholder="Password")

        col1, col2, col3 = st.columns(3)
        with col1:
            cancel = st.form_submit_button("Cancel")
        with col2:
            submit = st.form_submit_button("Login")
        with col3:
            register = st.form_submit_button("Register")

    if submit:
        # if password != confirm_password:
        #     st.error("Passwords do not match")
        # else:
            # Create a new database session
        session = Session()
        
        try:
            # check for user
            user = session.query(User).filter_by(email=email).first()
            
            if user:
                # User exists, check password
                if (user.password == password):  # Assuming passwords are hashed
                    st.session_state.user_id = user.id
                    st.success("Login successful! Click Login again to continue")
                    st.session_state.page = "home"
                else:
                    st.error("Incorrect password")
            else:
                st.error("User does not exist")
            # st.session_state.page = "home"
        except IntegrityError:
            st.error("A user with this email already exists")
            session.rollback()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            session.rollback()
        finally:
            session.close()
    
    if register:
        st.session_state.page = "register"
        st.success("Click again to proceed")

def home_page():
    if is_admin(st.session_state.user_id):
        # st.session_state.page = "admin"
        st.sidebar.button("Admin Dashboard", key="admin_dashboard", on_click=lambda: setattr(st.session_state, 'page', 'admin'))
    
    st.sidebar.button("Logout", key="logout", on_click=logout)
    # Add a button to the home_page() function to navigate to the gallery:
    st.sidebar.button("View Gallery", key="view_gallery", on_click=lambda: setattr(st.session_state, 'page', 'gallery'))
    st.sidebar.button("FeedBack", key="send_feedback", on_click=lambda: setattr(st.session_state, 'page', 'feedback'))

    st.title("A.B-Image-Generation")
    
    image_description = st.text_area("Describe an image you would like to generate", 
                                    placeholder="eg A woman sitting on top of a tree")
    
    num_images = st.slider("Number of images to generate", 1, 10, 1)
    styles = ["Minimalistic", "Futuristic", "Vintage", "Sketch", "Modern", "Retro"]

    # Create the dropdown
    selected_style = st.selectbox("Choose a style:", styles)
    
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            if selected_style:
                image_description = image_description + " Make it look " + selected_style
            session = Session()
            try:
                # Create new user 
                new_request = ImageRequest(
                    user_id = st.session_state.user_id,
                    description = image_description
                )
                session.add(new_request)
                session.commit()
                st.session_state.request_id = new_request.id
                st.success("Request submitted successfully")
                st.session_state.page = "home"
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                session.rollback()
            finally:
                session.close()

            generated = generate_image(image_description)
            image_data = generated['output'][0]
            # image_data = 'https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/24ccdc3d-e494-462a-9228-80e8ff898a77-0.png'
            
            if(image_data):
                image_url = image_data
                session = Session()
                try:
                    # Create new user 
                    new_request = GeneratedImages(
                        request_id = st.session_state.request_id,
                        imageUrl = image_url
                    )
                    session.add(new_request)
                    session.commit()
                    # st.success("Request submitted successfully")
                    st.session_state.page = "home"
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    session.rollback()
                finally:
                    session.close()
                st.image(image_url, width=300)
                # st.image(st.session_state.generated_image, use_column_width=True)
                response = requests.get(image_url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    # Provide a download button for the image
                    st.download_button(
                        label="Download Image",
                        data=byte_im,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
                else:
                    st.error("Failed to fetch the image from the URL.")
            else:
                image_url = "https://t3.ftcdn.net/jpg/02/15/06/84/240_F_215068406_S4qqbkegOu1Al2dQF3plwI3oWJvDWxYo.jpg"
                st.image(image_url, width=300)


def gallery_page():
    if 'user_id' not in st.session_state:
        st.error("Please log in to view your gallery")
        st.session_state.page = "login"
        return

    st.title("Your Image Gallery")
    st.sidebar.button("Back to Home", key="back_to_home", on_click=lambda: setattr(st.session_state, 'page', 'home'))

    session = Session()
    try:
        # Fetch all image requests for the current user
        user_requests = session.query(ImageRequest).filter_by(user_id=st.session_state.user_id).all()

        if not user_requests:
            st.info("You haven't generated any images yet.")
            return

        # Create a container for the gallery
        gallery_container = st.container()

        # Initialize list to store all images
        all_images = []

        # Fetch all images for the user
        for request in user_requests:
            generated_images = session.query(GeneratedImages).filter_by(request_id=request.id).all()
            for image in generated_images:
                all_images.append({
                    'url': image.imageUrl,
                    'description': request.description,
                    'id': image.id
                })

        # Create rows and columns
        for i in range(0, len(all_images), 3):
            with gallery_container:
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(all_images):
                        image = all_images[i + j]
                        with cols[j]:
                            st.image(image['url'], use_column_width=True)
                            st.caption(f"Prompt: {image['description'][:50]}...")  # Show first 50 chars of description
                            
                            # Add download button for each image
                            response = requests.get(image['url'])
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                buf = BytesIO()
                                img.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="Download",
                                    data=byte_im,
                                    file_name=f"image_{image['id']}.png",
                                    mime="image/png",
                                    key=f"download_{image['id']}"
                                )

    except Exception as e:
        st.error(f"An error occurred while fetching your images: {str(e)}")
    finally:
        session.close()

def feedback_page():
    st.sidebar.button("Back to Home", key="back_to_home", on_click=lambda: setattr(st.session_state, 'page', 'home'))
    st.header("Submit Feedback or Complaint")
    with st.form("feedback_form"):
        feedback_type = st.selectbox("Type", ["Feedback", "Complaint"])
        feedback_message = st.text_area("Your message", max_chars=500)
        submit_feedback = st.form_submit_button("Submit")

    if submit_feedback:
        if feedback_message.strip():  # Ensure the message is not empty
            session = Session()
            try:
                new_feedback = Feedback(
                    user_id=st.session_state.user_id,
                    message= feedback_type + " : " + feedback_message,
                    # type=feedback_type.lower(),
                    created_at=datetime.utcnow()
                )
                session.add(new_feedback)
                session.commit()
                st.success("Your feedback has been submitted successfully!")
            except Exception as e:
                st.error(f"An error occurred while submitting your feedback: {str(e)}")
                session.rollback()
            finally:
                session.close()
        else:
            st.warning("Please enter a message before submitting.")

def logout():
    # Logic to handle logout
    st.session_state.page = "login"
    st.session_state.clear()  # Clear session state data
    st.success("You have been logged out.")

def set_bg_hack(color):
    
    custom_css = f"""
    <style>
        .stApp {{
            background-color: {color};
        }}
        .stSidebar, .stSidebar > div:first-child {{
            background-color: {color};
        }}
        .stToolbar, .stToolbar > div:first-child {{
            background-color: {color};
        }}
        .stHeader, .stHeader > div:first-child {{
            background-color: {color};
        }}
        .stMarkdown, .stMarkdown > div:first-child {{
            background-color: transparent;
        }}
        .stSelectbox > div:first-child {{
            background-color: {color};
        }}
        .stTextInput > div:first-child {{
            background-color: {color};
        }}
        .stTextArea > div:first-child {{
            background-color: {color};
        }}
        .stButton > button:first-child {{
            background-color: {color};
            border: 1px solid #ccc;  /* Optional: adds a border to buttons */
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
def main():
    if "page" not in st.session_state:
        st.session_state.page = "login"
    # set_bg_hack('#abdbe3')
    if st.session_state.page == "register":
        registration_page()
    elif st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "gallery":
        gallery_page()
    elif st.session_state.page == "admin":
        admin_page()
    elif st.session_state.page == "feedback":
        feedback_page()
    
if __name__ == "__main__":
    main()