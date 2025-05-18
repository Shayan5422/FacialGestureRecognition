import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cv2
import dlib
from scipy.spatial import distance as dist # For EAR calculation
import time # for progress bar simulation and potentially camera loop

# Constants for detection (from eye_eyebrow_detector.py)
EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR = 0.15
CALIBRATION_FRAMES = 30 # Reduced for faster demo calibration
EAR_THRESHOLD = 0.20
DLIB_SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Display states (from eye_eyebrow_detector.py)
STATE_YES = "Yes"
STATE_NO = "No"
STATE_NORMAL = "Normal"
STATE_CALIBRATING = "Calibrating..."

# Landmark indices (from eye_eyebrow_detector.py)
(user_L_eye_indices_start, user_L_eye_indices_end) = (42, 48)
(user_R_eye_indices_start, user_R_eye_indices_end) = (36, 42)
user_L_eye_top_indices = [43, 44]
user_R_eye_top_indices = [37, 38]
user_L_eyebrow_y_calc_indices = range(23, 26)
user_R_eyebrow_y_calc_indices = range(18, 21)


# Initialize dlib's face detector and facial landmark predictor
# We'll initialize this inside the function or manage its state
# to avoid issues with Streamlit's rerun behavior.

# Stock photo URLs provided
FACIAL_RECOGNITION_IMAGES = [
    "https://pixabay.com/get/g12854d8ea8c029d2435717f123bb6b3afe5f218d14e94f3f1bd28aedaf46900b3c663fdca24e3e5ff97ed203a4ac97bdd34215b14df2f288e76f20602a81cb7d_1280.jpg",
    "https://pixabay.com/get/gf7f1fe0deb60c9c2217635915c6efdd85c3a35b943185d9d7c1b08ead1ec8f6d082af4bfe7a16759a66c38872d828da9c7d28f9ccd6ed4c243f50471537c072d_1280.jpg",
    "https://pixabay.com/get/g5226c742de43d538d1d4dd7e927224fb5be1b7f0f197f568dedc10336530b516cf9b2b3acc3128a4ea78a43ca348c8ce101234788ff131ed802e296e799ddc00_1280.jpg",
    "https://pixabay.com/get/g95d27127dde404c64753341780b8d8871f128bda7dfd5cc3ef287e4e838a1719fc91bc6c4bb24c52ef7cf27dad266a50d474142afe73e25f207ef9ef375c268e_1280.jpg"
]

AI_DATA_VIZ_IMAGES = [
    "https://pixabay.com/get/g155188879e1e171fb82c63d79b2963561b3a77f46ecb38053344fb6a1e236c2f406d66b1c3ae23260573869a9458daee7bfc00f37ef6840fce3a379da3d608e4_1280.jpg",
    "https://pixabay.com/get/g2620d81b6747dcda89657292ec3627897d7e61e906e76de11ecf6babedfcbe40aa0d0608950e1474795bc3a2abc67660ebc08977ba37da526834bec3cf342ba1_1280.jpg",
    "https://pixabay.com/get/ge8f809c48922d0dd956c8896157bd3ea8f606948d2ff72e507bad98b42b823e6409cc2923100bc91b15a499f72263fd8ca0f0949ac5ad2bbbb176f16e3dd0043_1280.jpg",
    "https://pixabay.com/get/g20331e7a18a7b2759056b7a9a73d20c34ff4f863ec4660535f9e5a1b15d3ad4b5b72bb07c385dd3ce154dc23b72fedd5c1eb9e2a4f2b335dfb17534d2b11d8e0_1280.jpg"
]

PRESENTATION_SLIDE_IMAGES = [
    "https://pixabay.com/get/gb57703b075295316bc1711f9701b18b84cfb89f469bb77f415392cc8986f922927cabc9afd50638f77ed51f53bcc62f423b96fbeb5f008abd1017db5b33e9e96_1280.jpg",
    "https://pixabay.com/get/gf4116a5ec8333a8a6bb33dcfe0baecc03580e6f7af95f2895880c9ec051479f3af002ecde96686e5fb6d3a860cf794fef532f27d373318317330932475a8b46c_1280.jpg"
]

def section_header(title):
    """Generate a section header with consistent styling"""
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)

def render_intro_section():
    """Render the introduction section of the presentation"""
    section_header("Introduction")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        # Facial Gesture Recognition
        
        Facial gesture recognition is an exciting field at the intersection of computer vision and artificial intelligence that focuses on identifying and interpreting human facial expressions and movements.
        
        This presentation explores a system that can:
        
        - Detect facial landmarks in real-time video
        - Track specific facial movements (eyes, eyebrows)
        - Classify gestures into meaningful actions
        - Respond to gestures with appropriate system actions
        
        Using a combination of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** architecture, this system achieves high accuracy in real-time environments.
        """)
    
    with col2:
        st.image(FACIAL_RECOGNITION_IMAGES[0], use_container_width=True)
        st.caption("Facial recognition technology")
    
    st.markdown("---")
    
    st.markdown("""
    ### Why Facial Gesture Recognition Matters
    
    Facial gestures provide a natural, intuitive way for humans to communicate with computers:
    
    - **Accessibility**: Enables computer control for people with mobility limitations
    - **Hands-free Interaction**: Useful in environments where hands are occupied or contaminated
    - **Enhanced User Experience**: Creates more natural human-computer interactions
    - **Safety Applications**: Driver drowsiness detection, attention monitoring
    """)

def render_objective_section():
    """Render the project objectives section"""
    section_header("Project Objective")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ## Primary Goal
        
        Create an intelligent system that automatically recognizes facial gestures from a video stream in real-time.
        
        ### Key Objectives
        
        1. **Real-time Processing**: Analyze video frames with minimal latency
        2. **Accurate Detection**: Precisely identify facial landmarks
        3. **Gesture Classification**: Correctly interpret facial movements
        4. **Responsive Output**: Provide immediate feedback based on detected gestures
        """)
        
        st.markdown("""
        ### Target Gestures
        
        The system focuses on recognizing the following facial gestures:
        
        - Eye movements (blinks, winks)
        - Eyebrow movements (raising, furrowing)
        - Normal/neutral state
        """)
    
    with col2:
        
        
        # Add an interactive element - demo selector
        st.markdown("### Interactive Demo")
        gesture_type = st.selectbox(
            "Select a gesture type to learn more",
            ["Eye Movements", "Eyebrow Movements", "Neutral State"]
        )
        
        if gesture_type == "Eye Movements":
            st.info("Eye movements like blinks and winks can be used for selection or confirmation actions.")
        elif gesture_type == "Eyebrow Movements":
            st.info("Eyebrow raising can indicate interest or be used as a trigger for specific actions.")
        elif gesture_type == "Neutral State":
            st.info("The neutral state serves as the baseline for detecting deviations that signal intentional gestures.")

def render_architecture_section():
    """Render the architecture and methodology section"""
    section_header("Architecture & Methodology")
    
    st.markdown("""
    ## CNN-LSTM Architecture
    
    The system employs a hybrid deep learning architecture combining:
    
    - **Convolutional Neural Networks (CNN)**: Extract spatial features from facial images
    - **Long Short-Term Memory (LSTM)**: Capture temporal patterns in sequential frames
    """)
    
    # Display CNN-LSTM architecture diagram
    
    st.caption("Visual representation of CNN-LSTM architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### CNN Component
        
        The CNN portion of the architecture:
        
        - Processes individual video frames
        - Extracts spatial features from facial regions
        - Identifies key patterns in facial structure
        - Uses multiple convolutional layers with pooling
        """)
        
        # Create interactive CNN visualization
        st.markdown("#### CNN Layer Visualization")
        layer_slider = st.slider("Explore CNN layers", 1, 5, 1)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.title(f"CNN Layer {layer_slider} Feature Maps")
        
        # Generate mock feature map visualization
        grid_size = 4
        feature_maps = np.random.rand(grid_size, grid_size, 9)
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(feature_maps[:,:,i], cmap='viridis')
            plt.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        ### LSTM Component
        
        The LSTM network:
        
        - Processes sequences of CNN-extracted features
        - Captures temporal dependencies between frames
        - Maintains memory of previous facial states
        - Enables detection of dynamic gestures over time
        """)
        
        # Add interactive LSTM cell visualization
        st.markdown("#### LSTM Cell Structure")
        
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/LSTM_Cell.svg", caption="LSTM Cell Structure", use_container_width=True)
    
    st.markdown("""
    ### Combined Model Benefits
    
    This hybrid architecture provides several advantages:
    
    1. **Spatial-Temporal Processing**: Captures both spatial features and temporal patterns
    2. **Sequence Understanding**: Recognizes gestures that develop over multiple frames
    3. **Contextual Awareness**: Considers the progression of facial movements
    4. **Robust Classification**: Higher accuracy for dynamic gestures
    """)

def render_process_section():
    """Render the process flow section"""
    section_header("Process Flow")
    
    st.markdown("""
    ## System Workflow
    
    The facial gesture recognition process follows these key steps:
    """)
    
    # Create tabs for different stages of the process
    tab1, tab2, tab3 = st.tabs(["Data Collection", "Image Processing", "Model Training"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Data Collection
            
            The system requires a comprehensive dataset of facial gestures:
            
            - **Video Capture**: Short video clips recorded using webcam
            - **Gesture Performance**: Subjects perform predefined facial gestures
            - **Labeling**: Each video is labeled with the corresponding gesture
            - **Dataset Diversity**: Multiple subjects, lighting conditions, and angles
            
            A balanced dataset with various examples of each gesture is crucial for model generalization.
            """)
        
        with col2:
            
            st.caption("")
    
    with tab2:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.image(AI_DATA_VIZ_IMAGES[0], use_container_width=True)
            st.caption("Image processing visualization")
        
        with col2:
            st.markdown("""
            ### Image Processing
            
            Raw video frames undergo several preprocessing steps:
            
            1. **Facial Detection**: Locating the face in each frame
            2. **Landmark Extraction**: Identifying 68 key facial points
            3. **Region Isolation**: Extracting regions of interest (eyes, eyebrows)
            4. **Normalization**: Converting to grayscale, normalizing pixel values
            5. **Augmentation**: Generating additional training samples through transformations
            
            These steps ensure the input data is optimized for the neural network.
            """)
            
            # Interactive element - landmark detection demo
            show_landmarks = st.checkbox("Show facial landmarks example (eyes and eyebrows)")
            if show_landmarks:
                landmark_cols = st.columns(2)
                with landmark_cols[0]:
                    # Mock landmark visualization using matplotlib - focusing on eyes and eyebrows
                    fig, ax = plt.subplots(figsize=(4, 4))
                    
                    # Create a simple face outline
                    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='blue')
                    ax.add_patch(circle)
                    
                    # Add eye landmarks with extra detail (6 points per eye)
                    # Left eye
                    left_eye_x = [0.30, 0.33, 0.37, 0.41, 0.38, 0.34]
                    left_eye_y = [0.60, 0.58, 0.58, 0.60, 0.62, 0.62]
                    ax.plot(left_eye_x, left_eye_y, 'g-', linewidth=2)
                    for x, y in zip(left_eye_x, left_eye_y):
                        ax.plot(x, y, 'go', markersize=4)
                    
                    # Right eye
                    right_eye_x = [0.59, 0.62, 0.66, 0.70, 0.67, 0.63]
                    right_eye_y = [0.60, 0.58, 0.58, 0.60, 0.62, 0.62]
                    ax.plot(right_eye_x, right_eye_y, 'g-', linewidth=2)
                    for x, y in zip(right_eye_x, right_eye_y):
                        ax.plot(x, y, 'go', markersize=4)
                    
                    # Add detailed eyebrow landmarks (5 points per eyebrow)
                    # Left eyebrow
                    left_brow_x = [0.25, 0.30, 0.35, 0.40, 0.45]
                    left_brow_y = [0.70, 0.72, 0.73, 0.72, 0.70]
                    ax.plot(left_brow_x, left_brow_y, 'r-', linewidth=2)
                    for x, y in zip(left_brow_x, left_brow_y):
                        ax.plot(x, y, 'ro', markersize=4)
                    
                    # Right eyebrow
                    right_brow_x = [0.55, 0.60, 0.65, 0.70, 0.75]
                    right_brow_y = [0.70, 0.72, 0.73, 0.72, 0.70]
                    ax.plot(right_brow_x, right_brow_y, 'r-', linewidth=2)
                    for x, y in zip(right_brow_x, right_brow_y):
                        ax.plot(x, y, 'ro', markersize=4)
                    
                    # Add labels
                    ax.text(0.36, 0.67, "Left Eye", fontsize=9, ha='center')
                    ax.text(0.64, 0.67, "Right Eye", fontsize=9, ha='center')
                    ax.text(0.35, 0.76, "Left Eyebrow", fontsize=9, ha='center')
                    ax.text(0.65, 0.76, "Right Eyebrow", fontsize=9, ha='center')
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_title("Eye and Eyebrow Landmarks")
                    ax.axis('off')
                    
                    st.pyplot(fig)
                
                with landmark_cols[1]:
                    st.markdown("""
                    **Focused Facial Landmarks Analysis:**
                    
                    This system specifically analyzes:
                    
                    - **Eyes (6 points each)**: Tracks eye openness, blinks, and winking
                    - **Eyebrows (5 points each)**: Detects eyebrow raising, furrowing, and expressions
                    
                    While the shape_predictor_68_face_landmarks model can identify 68 facial landmarks including:
                    - 9 points for the nose
                    - 20 points for the mouth
                    - 17 points for the face contour
                    
                    This implementation focuses exclusively on eye and eyebrow movements for gesture recognition.
                    """)
    
    with tab3:
        st.markdown("""
        ### Model Training
        
        The CNN-LSTM model is trained using the processed dataset:
        
        1. **Data Splitting**: Division into training, validation, and test sets
        2. **CNN Training**: Learning spatial feature extraction
        3. **LSTM Training**: Learning temporal patterns
        4. **Hyperparameter Tuning**: Optimizing model architecture and parameters
        5. **Validation**: Evaluating performance on validation set
        6. **Testing**: Final evaluation on test set
        """)
        
        # Interactive training visualization
        st.markdown("#### Training Visualization")
        
        # Mock training metrics
        epochs = 50
        train_loss = 1.5 * np.exp(-0.05 * np.arange(epochs)) + 0.1 * np.random.rand(epochs)
        val_loss = 1.7 * np.exp(-0.04 * np.arange(epochs)) + 0.15 * np.random.rand(epochs)
        train_acc = 1 - train_loss * 0.5
        val_acc = 1 - val_loss * 0.5
        
        # Create interactive plot
        metric = st.radio("Select metric to visualize", ["Loss", "Accuracy"])
        
        if metric == "Loss":
            fig = px.line(
                x=list(range(1, epochs+1)),
                y=[train_loss, val_loss],
                labels={"x": "Epoch", "y": "Loss"},
                title="Training and Validation Loss",
                line_shape="spline"
            )
            fig.update_layout(legend_title_text="Legend")
            fig.add_scatter(x=list(range(1, epochs+1)), y=train_loss, name="Training Loss", line=dict(color="blue"))
            fig.add_scatter(x=list(range(1, epochs+1)), y=val_loss, name="Validation Loss", line=dict(color="red"))
        else:
            fig = px.line(
                x=list(range(1, epochs+1)),
                y=[train_acc, val_acc],
                labels={"x": "Epoch", "y": "Accuracy"},
                title="Training and Validation Accuracy",
                line_shape="spline"
            )
            fig.update_layout(legend_title_text="Legend")
            fig.add_scatter(x=list(range(1, epochs+1)), y=train_acc, name="Training Accuracy", line=dict(color="green"))
            fig.add_scatter(x=list(range(1, epochs+1)), y=val_acc, name="Validation Accuracy", line=dict(color="orange"))
        
        st.plotly_chart(fig)
    
    

def render_technology_section():
    """Render the technologies section"""
    section_header("Technologies")
    
    st.markdown("""
    ## Core Technologies
    
    The facial gesture recognition system relies on several key technologies:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Python Ecosystem
        
        - **Python**: Core programming language
        - **NumPy**: Numerical operations
        - **Pandas**: Data management
        - **Matplotlib/Plotly**: Visualization
        """)
        
        st.image(AI_DATA_VIZ_IMAGES[2], use_container_width=True)
        st.caption("Python data analysis visualization")
    
    with col2:
        st.markdown("""
        ### Deep Learning
        
        - **TensorFlow/Keras**: Neural network framework
        - **CNN**: Spatial feature extraction
        - **LSTM**: Temporal sequence processing
        - **Transfer Learning**: Leveraging pre-trained models
        """)
        
        # Create an interactive visualization of model architecture
        st.markdown("#### Model Architecture")
        
        fig = go.Figure()
        
        # Draw rectangles representing layers
        layers = [
            {"name": "Input", "width": 0.8, "height": 0.15, "x": 0.5, "y": 0.9, "color": "lightblue"},
            {"name": "Conv2D", "width": 0.8, "height": 0.1, "x": 0.5, "y": 0.75, "color": "lightgreen"},
            {"name": "MaxPooling", "width": 0.7, "height": 0.1, "x": 0.5, "y": 0.63, "color": "lightgreen"},
            {"name": "Conv2D", "width": 0.6, "height": 0.1, "x": 0.5, "y": 0.51, "color": "lightgreen"},
            {"name": "MaxPooling", "width": 0.5, "height": 0.1, "x": 0.5, "y": 0.39, "color": "lightgreen"},
            {"name": "LSTM", "width": 0.8, "height": 0.1, "x": 0.5, "y": 0.27, "color": "lightpink"},
            {"name": "Dense", "width": 0.6, "height": 0.1, "x": 0.5, "y": 0.15, "color": "lightyellow"},
            {"name": "Output", "width": 0.4, "height": 0.1, "x": 0.5, "y": 0.05, "color": "lightblue"}
        ]
        
        for layer in layers:
            # Add layer rectangle
            fig.add_shape(
                type="rect",
                x0=layer["x"] - layer["width"]/2,
                y0=layer["y"] - layer["height"]/2,
                x1=layer["x"] + layer["width"]/2,
                y1=layer["y"] + layer["height"]/2,
                line=dict(color="black"),
                fillcolor=layer["color"]
            )
            
            # Add layer name
            fig.add_annotation(
                x=layer["x"],
                y=layer["y"],
                text=layer["name"],
                showarrow=False
            )
            
            # Add connection lines between layers (except for the last layer)
            if layer != layers[-1]:
                next_layer = layers[layers.index(layer) + 1]
                fig.add_shape(
                    type="line",
                    x0=layer["x"],
                    y0=layer["y"] - layer["height"]/2,
                    x1=next_layer["x"],
                    y1=next_layer["y"] + next_layer["height"]/2,
                    line=dict(color="gray", width=1)
                )
        
        fig.update_layout(
            showlegend=False,
            width=300,
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1])
        )
        
        st.plotly_chart(fig)
    
    with col3:
        st.markdown("""
        ### Computer Vision
        
        - **OpenCV**: Image and video processing
        - **Dlib**: Facial landmark detection
        - **MediaPipe**: Real-time face mesh tracking
        - **Image augmentation**: Diverse training samples
        """)
        
        st.image(AI_DATA_VIZ_IMAGES[3], use_container_width=True)
        st.caption("Computer vision analysis")
    
    st.markdown("---")
    
    # Technology performance comparison
    st.markdown("### Performance Comparison")
    
    # Create a mock performance comparison chart
    performance_data = {
        'Method': ['Traditional CV', 'CNN Only', 'LSTM Only', 'CNN-LSTM'],
        'Accuracy': [65, 82, 78, 93],
        'Speed (FPS)': [45, 28, 32, 25],
        'Memory (MB)': [120, 350, 280, 420]
    }
    
    metric_to_show = st.selectbox("Select performance metric", ["Accuracy", "Speed (FPS)", "Memory (MB)"])
    
    fig = px.bar(
        performance_data,
        x='Method',
        y=metric_to_show,
        color='Method',
        text=performance_data[metric_to_show],
        title=f"Performance Comparison - {metric_to_show}"
    )
    
    # Customize the chart appearance
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig)

def render_applications_section():
    """Render the applications section"""
    section_header("Applications")
    
    st.markdown("""
    ## Potential Applications
    
    Facial gesture recognition technology has numerous practical applications across various domains:
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        
        
        st.markdown("""
        ### Human-Computer Interaction
        
        - **Hands-free Computing**: Control computers without physical input devices
        - **Accessible Technology**: Enable computer usage for people with mobility limitations
        - **Interactive Presentations**: Control slides and demonstrations with facial gestures
        - **Gaming**: Enhanced immersion through facial expression controls
        """)
        
        st.markdown("""
        ### Healthcare
        
        - **Patient Monitoring**: Track patient attentiveness or consciousness
        - **Rehabilitation**: Provide feedback for facial exercises
        - **Pain Assessment**: Detect discomfort through facial expressions
        - **Mental Health**: Analyze emotional responses during therapy
        """)
    
    with col2:
        st.markdown("""
        ### Automotive Applications
        
        - **Driver Monitoring**: Detect drowsiness or distraction
        - **In-car Controls**: Adjust settings with facial gestures
        - **Personalized Experience**: Recognize driver identity and preferences
        - **Security**: Additional authentication layer
        """)
        
        st.markdown("""
        ### Accessibility
        
        - **Assistive Technology**: Enable computer control for users with mobility impairments
        - **Communication Aids**: Help non-verbal individuals express themselves
        - **Smart Home Control**: Manage home automation with facial gestures
        - **Public Kiosks**: Enable gesture-based interaction with public information systems
        """)
        
        
    
    # Interactive application explorer
    st.markdown("### Application Explorer")
    
    application_area = st.selectbox(
        "Select an application area to explore",
        ["Human-Computer Interaction", "Healthcare", "Automotive", "Accessibility", "Education"]
    )
    
    if application_area == "Human-Computer Interaction":
        st.info("""
        **Featured Application: Gesture-Controlled Presentation System**
        
        A system that allows presenters to control slideshows using facial gestures:
        - Eye blinks to advance slides
        - Eyebrow raises to go back
        - Head nods/shakes to confirm/cancel actions
        
        This enables hands-free presentations, allowing speakers to maintain natural gestures while speaking.
        """)
    elif application_area == "Healthcare":
        st.info("""
        **Featured Application: Pain Assessment Tool**
        
        A system that monitors patient facial expressions to detect signs of pain:
        - Real-time monitoring without requiring verbal communication
        - Particularly useful for non-verbal patients or those with cognitive impairments
        - Alerts medical staff when pain indicators are detected
        - Maintains a log of pain expression events for medical review
        """)
    elif application_area == "Automotive":
        st.info("""
        **Featured Application: Driver Alertness Monitoring**
        
        A system that detects signs of driver fatigue or distraction:
        - Monitors eye closure duration and blink rate
        - Detects head nodding indicative of drowsiness
        - Provides audio alerts when fatigue signs are detected
        - Suggests breaks when sustained fatigue patterns are observed
        """)
    elif application_area == "Accessibility":
        st.info("""
        **Featured Application: Facial Gesture Computer Control**
        
        A complete computer control system for people with limited mobility:
        - Cursor movement through slight head movements
        - Selection through eye blinks or eyebrow raises
        - Scrolling through specific eye movements
        - Text input through an on-screen keyboard navigated by facial gestures
        """)
    elif application_area == "Education":
        st.info("""
        **Featured Application: Student Engagement Analytics**
        
        A system that monitors student facial expressions during online learning:
        - Tracks attentiveness and engagement through eye movements
        - Identifies confusion through facial expressions
        - Provides analytics to instructors about student engagement
        - Helps identify content that may need additional explanation
        """)
    
    # Conclusion
    st.markdown("---")
    st.markdown("""
    ## Conclusion
    
    Facial gesture recognition using AI represents a significant advancement in human-computer interaction. By combining CNN and LSTM architectures, we've created a system that can:
    
    - Accurately recognize facial gestures in real-time
    - Process video streams with minimal latency
    - Adapt to different users and environments
    - Enable new possibilities for accessibility and interaction
    
    This technology continues to evolve, with ongoing improvements in accuracy, speed, and adaptability.
    """)
    
    st.success("Thank you for exploring this presentation on Facial Gesture Recognition using AI!")

    # Using the SVG file from assets instead of embedding directly
    st.image("assets/workflow_diagram.svg")

def get_landmark_point_from_detector(landmarks, index):
    """Helper function from eye_eyebrow_detector.py"""
    return (landmarks.part(index).x, landmarks.part(index).y)

def eye_aspect_ratio_from_detector(eye_pts):
    """Helper function from eye_eyebrow_detector.py"""
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    ear_val = (A + B) / (2.0 * C)
    return ear_val

def initialize_dlib_components():
    """Initializes dlib detector and predictor."""
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_SHAPE_PREDICTOR_PATH)
        return detector, predictor
    except RuntimeError as e:
        st.error(f"Failed to load dlib model: {e}. Please ensure '{DLIB_SHAPE_PREDICTOR_PATH}' is in the correct path.")
        return None, None

def render_live_demo_section():
    """Render the live facial gesture recognition demo section"""
    section_header("Live Facial Gesture Demo")
    st.write("This demo uses your webcam to perform real-time eye and eyebrow gesture detection.")
    st.warning("Ensure you have a webcam connected and have granted permission if prompted by your browser. Also, make sure `shape_predictor_68_face_landmarks.dat` is in the application's root directory.")

    if 'detector' not in st.session_state or 'predictor' not in st.session_state:
        st.session_state.detector, st.session_state.predictor = initialize_dlib_components()

    if st.session_state.detector is None or st.session_state.predictor is None:
        st.error("Dlib components could not be initialized. The demo cannot run.")
        return

    # Initialize session state variables for the demo
    if 'run_demo' not in st.session_state:
        st.session_state.run_demo = False
    if 'calibration_counter' not in st.session_state:
        st.session_state.calibration_counter = 0
    if 'calibration_data_user_L_eyebrow_y' not in st.session_state:
        st.session_state.calibration_data_user_L_eyebrow_y = []
    if 'calibration_data_user_R_eyebrow_y' not in st.session_state:
        st.session_state.calibration_data_user_R_eyebrow_y = []
    if 'calibration_data_user_L_eye_top_y' not in st.session_state:
        st.session_state.calibration_data_user_L_eye_top_y = []
    if 'calibration_data_user_R_eye_top_y' not in st.session_state:
        st.session_state.calibration_data_user_R_eye_top_y = []
    if 'normal_user_L_eyebrow_y_avg' not in st.session_state:
        st.session_state.normal_user_L_eyebrow_y_avg = 0
    if 'normal_user_R_eyebrow_y_avg' not in st.session_state:
        st.session_state.normal_user_R_eyebrow_y_avg = 0
    if 'normal_user_L_eye_top_y_avg' not in st.session_state:
        st.session_state.normal_user_L_eye_top_y_avg = 0
    if 'normal_user_R_eye_top_y_avg' not in st.session_state:
        st.session_state.normal_user_R_eye_top_y_avg = 0
    if 'normal_dist_L_eyebrow_to_eye' not in st.session_state:
        st.session_state.normal_dist_L_eyebrow_to_eye = 0
    if 'normal_dist_R_eyebrow_to_eye' not in st.session_state:
        st.session_state.normal_dist_R_eyebrow_to_eye = 0
    if 'current_state_demo' not in st.session_state:
        st.session_state.current_state_demo = STATE_CALIBRATING
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False


    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start/Restart Demo"):
            st.session_state.run_demo = True
            st.session_state.camera_active = True
            # Reset calibration
            st.session_state.calibration_counter = 0
            st.session_state.calibration_data_user_L_eyebrow_y = []
            st.session_state.calibration_data_user_R_eyebrow_y = []
            st.session_state.calibration_data_user_L_eye_top_y = []
            st.session_state.calibration_data_user_R_eye_top_y = []
            st.session_state.current_state_demo = STATE_CALIBRATING
            st.info("Calibration started. Look at the camera with a normal expression.")
    with col2:
        if st.button("Stop Demo"):
            st.session_state.run_demo = False
            st.session_state.camera_active = False


    if st.session_state.run_demo and st.session_state.camera_active:
        # Placeholder for video feed
        frame_placeholder = st.empty()
        
        # Attempt to open the webcam
        # We manage cap in session_state to persist it across reruns if needed,
        # but for a continuous loop, it's tricky.
        # A common pattern is to release it if we stop.
        if 'cap' not in st.session_state or not st.session_state.cap.isOpened():
             st.session_state.cap = cv2.VideoCapture(0)

        if not st.session_state.cap.isOpened():
            st.error("Cannot open webcam.")
            st.session_state.run_demo = False # Stop demo if camera fails
            return

        detector = st.session_state.detector
        predictor = st.session_state.predictor

        while st.session_state.run_demo and st.session_state.cap.isOpened():
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("Failed to grab frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            display_text = st.session_state.current_state_demo
            
            if st.session_state.calibration_counter < CALIBRATION_FRAMES:
                st.session_state.current_state_demo = STATE_CALIBRATING
                display_text = f"{STATE_CALIBRATING} ({st.session_state.calibration_counter}/{CALIBRATION_FRAMES})"


            for face in faces:
                landmarks = predictor(gray, face)

                user_L_eyebrow_current_y_pts = [landmarks.part(i).y for i in user_L_eyebrow_y_calc_indices]
                current_user_L_eyebrow_y_avg = np.mean(user_L_eyebrow_current_y_pts) if user_L_eyebrow_current_y_pts else 0

                user_R_eyebrow_current_y_pts = [landmarks.part(i).y for i in user_R_eyebrow_y_calc_indices]
                current_user_R_eyebrow_y_avg = np.mean(user_R_eyebrow_current_y_pts) if user_R_eyebrow_current_y_pts else 0

                user_L_eye_top_current_y_pts = [landmarks.part(i).y for i in user_L_eye_top_indices]
                current_user_L_eye_top_y_avg = np.mean(user_L_eye_top_current_y_pts) if user_L_eye_top_current_y_pts else 0
                
                user_R_eye_top_current_y_pts = [landmarks.part(i).y for i in user_R_eye_top_indices]
                current_user_R_eye_top_y_avg = np.mean(user_R_eye_top_current_y_pts) if user_R_eye_top_current_y_pts else 0

                user_L_eye_all_pts = np.array([get_landmark_point_from_detector(landmarks, i) for i in range(user_L_eye_indices_start, user_L_eye_indices_end)], dtype="int")
                user_R_eye_all_pts = np.array([get_landmark_point_from_detector(landmarks, i) for i in range(user_R_eye_indices_start, user_R_eye_indices_end)], dtype="int")
                
                left_ear = eye_aspect_ratio_from_detector(user_L_eye_all_pts)
                right_ear = eye_aspect_ratio_from_detector(user_R_eye_all_pts)
                avg_ear = (left_ear + right_ear) / 2.0

                if st.session_state.calibration_counter < CALIBRATION_FRAMES:
                    st.session_state.calibration_data_user_L_eyebrow_y.append(current_user_L_eyebrow_y_avg)
                    st.session_state.calibration_data_user_R_eyebrow_y.append(current_user_R_eyebrow_y_avg)
                    st.session_state.calibration_data_user_L_eye_top_y.append(current_user_L_eye_top_y_avg)
                    st.session_state.calibration_data_user_R_eye_top_y.append(current_user_R_eye_top_y_avg)
                    st.session_state.calibration_counter += 1
                    
                    display_text = f"{STATE_CALIBRATING} ({st.session_state.calibration_counter}/{CALIBRATION_FRAMES})"

                    if st.session_state.calibration_counter == CALIBRATION_FRAMES:
                        st.session_state.normal_user_L_eyebrow_y_avg = np.mean(st.session_state.calibration_data_user_L_eyebrow_y) if st.session_state.calibration_data_user_L_eyebrow_y else 0
                        st.session_state.normal_user_R_eyebrow_y_avg = np.mean(st.session_state.calibration_data_user_R_eyebrow_y) if st.session_state.calibration_data_user_R_eyebrow_y else 0
                        st.session_state.normal_user_L_eye_top_y_avg = np.mean(st.session_state.calibration_data_user_L_eye_top_y) if st.session_state.calibration_data_user_L_eye_top_y else 0
                        st.session_state.normal_user_R_eye_top_y_avg = np.mean(st.session_state.calibration_data_user_R_eye_top_y) if st.session_state.calibration_data_user_R_eye_top_y else 0

                        st.session_state.normal_dist_L_eyebrow_to_eye = st.session_state.normal_user_L_eye_top_y_avg - st.session_state.normal_user_L_eyebrow_y_avg
                        st.session_state.normal_dist_R_eyebrow_to_eye = st.session_state.normal_user_R_eye_top_y_avg - st.session_state.normal_user_R_eyebrow_y_avg
                        
                        st.session_state.current_state_demo = STATE_NORMAL
                        display_text = STATE_NORMAL
                        st.success("Calibration finished.")
                else: # Detection Phase
                    st.session_state.current_state_demo = STATE_NORMAL # Default to normal after calibration
                    display_text = STATE_NORMAL
                    if st.session_state.normal_dist_L_eyebrow_to_eye != 0 and st.session_state.normal_dist_R_eyebrow_to_eye != 0:
                        if avg_ear < EAR_THRESHOLD:
                            st.session_state.current_state_demo = STATE_YES
                            display_text = STATE_YES
                        else:
                            current_dist_L = current_user_L_eye_top_y_avg - current_user_L_eyebrow_y_avg
                            current_dist_R = current_user_R_eye_top_y_avg - current_user_R_eyebrow_y_avg

                            threshold_dist_L = st.session_state.normal_dist_L_eyebrow_to_eye * (1 + EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR)
                            threshold_dist_R = st.session_state.normal_dist_R_eyebrow_to_eye * (1 + EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR)
                            
                            if st.session_state.normal_dist_L_eyebrow_to_eye <= 0: threshold_dist_L = st.session_state.normal_dist_L_eyebrow_to_eye + abs(st.session_state.normal_dist_L_eyebrow_to_eye * EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR) + 5
                            if st.session_state.normal_dist_R_eyebrow_to_eye <= 0: threshold_dist_R = st.session_state.normal_dist_R_eyebrow_to_eye + abs(st.session_state.normal_dist_R_eyebrow_to_eye * EYEBROW_TO_EYE_VERTICAL_DISTANCE_INCREASE_FACTOR) + 5

                            if current_dist_L > threshold_dist_L and current_dist_R > threshold_dist_R:
                                st.session_state.current_state_demo = STATE_NO
                                display_text = STATE_NO
            
            # Display the detected state on the frame
            color = (255, 255, 0) # Default for Normal/Calibrating
            if st.session_state.current_state_demo == STATE_YES:
                color = (0, 255, 0)
            elif st.session_state.current_state_demo == STATE_NO:
                color = (0, 0, 255)
            
            # Make text larger and position it higher
            cv2.putText(frame, display_text, (frame.shape[1] // 2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            
            # Add a small delay to make the video smoother and allow Streamlit to process
            # time.sleep(0.01) # Removed for faster processing, relying on inherent delays

        # Release camera when demo stops or an error occurs
        if 'cap' in st.session_state and st.session_state.cap.isOpened():
            st.session_state.cap.release()
        if st.session_state.camera_active is False and 'cap' in st.session_state: # if explicitly stopped
             del st.session_state.cap


    elif not st.session_state.run_demo and st.session_state.camera_active:
        # This case handles when Stop Demo is clicked, ensuring camera is released.
        if 'cap' in st.session_state and st.session_state.cap.isOpened():
            st.session_state.cap.release()
            del st.session_state.cap # Ensure it's re-initialized if started again
        st.session_state.camera_active = False
        st.info("Live demo stopped.")

# Example of how to call this new section in a main app structure:
# if __name__ == "__main__":
#     st.set_page_config(layout="wide")
#     # Apply custom CSS (optional)
#     # st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
#
#     render_intro_section()
#     render_objective_section()
#     render_architecture_section()
#     render_process_section()
#     render_technology_section()
#     render_applications_section()
#     render_live_demo_section() # New section added here
#
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Introduction", "Objective", "Architecture", "Process Flow", "Technologies", "Applications", "Live Demo"])
#
#     if page == "Introduction": render_intro_section()
#     elif page == "Objective": render_objective_section()
#     # ... etc. for other sections
#     elif page == "Live Demo": render_live_demo_section() # Call if selected from sidebar too.
#     # This part is just an example of how one might structure the main app.
#     # The key is that `render_live_demo_section()` can now be called.
