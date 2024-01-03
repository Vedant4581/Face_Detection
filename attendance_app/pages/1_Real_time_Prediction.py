# #Real time prediction


from Home import st
from Home import face_recognition
from PIL import Image
from Home import pd
from Home import cv2
from insightface.app import FaceAnalysis

faceapp = FaceAnalysis(name='buffalo_l', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    img.save("i.jpg")
    return img

st.subheader('Real-Time Attendance System')

# Retrieve data from redis database
with st.spinner("Retrieving Data from Redis db..."):
    redis_face_db = face_recognition.retrieve_data(name='class:register')
    st.dataframe(redis_face_db)
    abc = redis_face_db.copy()

st.success("Data successfully retrieved from db")

selected_date = st.date_input('Select a date')
st.write('You selected:', selected_date)

# Upload the image
image_file = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"])
if image_file is not None:
    img = load_image(image_file)
    st.image(img, width=250)
    datacopyframe = redis_face_db.copy()
    t = cv2.imread("i.jpg")
    # st.write(t)  # Checking if the image is loaded correctly

    results = faceapp.get(t)
    test_copy = t.copy()
    xyz = []

    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']
        person_name, person_roll = face_recognition.ml_search_algorithm(datacopyframe, 'facial features', test_vector=embeddings, name_roll=['Name', 'Roll'], thresh=0.5)
        cv2.rectangle(test_copy, (x1, y1), (x2, y2), (0, 255, 0))
        text_gen = person_name
        if person_name == 'Unknown':
            cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 0, 255), 1)
        else:
            cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 255, 0), 1)
            text_hide=person_roll
            xyz.append(text_hide)

    st.image(test_copy, channels="BGR")  # Display the image with the detected faces
    st.write(xyz)  # Display the names of the recognized individuals


    

    p = datacopyframe.copy()

    # q=pd.read_csv('current_df.csv')
    try:
        q=pd.read_csv(f'{selected_date}.csv')
    except:
        q=pd.DataFrame(p)
        q.drop(['facial features'],axis=1)
    # Sorting the DataFrame by the 'Roll' column
    q = q.sort_values(by=['Roll'])
    q=q.reset_index(drop=True)
    q.set_index('Roll', inplace=True)
      # Displaying the DataFrame after the operations
    if selected_date in q.columns:
        for student in xyz:
            if q.at[student,selected_date]=="P":
                pass
            q.at[student,selected_date]="P"
    else:
        q[selected_date]="A"
        for student in xyz:
            q.at[student,selected_date]="P"

    q.to_csv(f'{selected_date}.csv')
    st.write(q)
