import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
        - Track specific facial movements (eyes, eyebrows, mouth)
        - Classify gestures into meaningful actions
        - Respond to gestures with appropriate system actions
        
        Using a combination of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** architecture, this system achieves high accuracy in real-time environments.
        """)
    
    with col2:
        st.image(FACIAL_RECOGNITION_IMAGES[0], use_column_width=True)
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
        - Head gestures (nodding yes, shaking no)
        - Normal/neutral state
        """)
    
    with col2:
        st.image(FACIAL_RECOGNITION_IMAGES[1], use_column_width=True)
        st.caption("Facial recognition technology in action")
        
        # Add an interactive element - demo selector
        st.markdown("### Interactive Demo")
        gesture_type = st.selectbox(
            "Select a gesture type to learn more",
            ["Eye Movements", "Eyebrow Movements", "Head Gestures", "Neutral State"]
        )
        
        if gesture_type == "Eye Movements":
            st.info("Eye movements like blinks and winks can be used for selection or confirmation actions.")
        elif gesture_type == "Eyebrow Movements":
            st.info("Eyebrow raising can indicate interest or be used as a trigger for specific actions.")
        elif gesture_type == "Head Gestures":
            st.info("Nodding (yes) and shaking (no) provide intuitive ways to confirm or reject prompts.")
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
    st.image(AI_DATA_VIZ_IMAGES[1], use_column_width=True)
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
        
        # Using Plotly for interactive visualization
        fig = go.Figure()
        
        # Draw the LSTM cell components
        fig.add_shape(type="rect", x0=0.1, y0=0.1, x1=0.9, y1=0.9, 
                    line=dict(color="RoyalBlue"), fillcolor="lightblue", opacity=0.3)
        
        # Input gate
        fig.add_shape(type="circle", x0=0.2, y0=0.6, x1=0.3, y1=0.7, 
                    line=dict(color="green"), fillcolor="lightgreen")
        
        # Forget gate
        fig.add_shape(type="circle", x0=0.2, y0=0.4, x1=0.3, y1=0.5, 
                    line=dict(color="red"), fillcolor="lightpink")
        
        # Output gate
        fig.add_shape(type="circle", x0=0.7, y0=0.5, x1=0.8, y1=0.6, 
                    line=dict(color="purple"), fillcolor="lavender")
        
        # Cell state line
        fig.add_shape(type="line", x0=0.1, y0=0.5, x1=0.9, y1=0.5, 
                    line=dict(color="black", width=2))
        
        # Add annotations
        fig.add_annotation(x=0.25, y=0.65, text="Input<br>Gate", showarrow=False)
        fig.add_annotation(x=0.25, y=0.45, text="Forget<br>Gate", showarrow=False)
        fig.add_annotation(x=0.75, y=0.55, text="Output<br>Gate", showarrow=False)
        fig.add_annotation(x=0.5, y=0.8, text="Cell State", showarrow=False)
        
        fig.update_layout(
            title="LSTM Cell Structure",
            showlegend=False,
            width=400,
            height=300,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        st.plotly_chart(fig)
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["Data Collection", "Image Processing", "Model Training", "Real-time Prediction"])
    
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
            st.image(FACIAL_RECOGNITION_IMAGES[2], use_column_width=True)
            st.caption("Data collection process")
    
    with tab2:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.image(AI_DATA_VIZ_IMAGES[0], use_column_width=True)
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
            show_landmarks = st.checkbox("Show facial landmarks example")
            if show_landmarks:
                landmark_cols = st.columns(2)
                with landmark_cols[0]:
                    # Mock landmark visualization using matplotlib
                    fig, ax = plt.subplots(figsize=(4, 4))
                    
                    # Create a simple face outline
                    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='blue')
                    ax.add_patch(circle)
                    
                    # Add mock landmarks
                    # Eyes
                    for x, y in [(0.35, 0.6), (0.65, 0.6)]:
                        eye = plt.Circle((x, y), 0.05, fill=False, color='green')
                        ax.add_patch(eye)
                        ax.plot(x, y, 'ro', markersize=3)
                    
                    # Eyebrows
                    ax.plot([0.25, 0.45], [0.7, 0.7], 'r-')
                    ax.plot([0.55, 0.75], [0.7, 0.7], 'r-')
                    
                    # Nose
                    ax.plot(0.5, 0.5, 'ro', markersize=3)
                    ax.plot([0.45, 0.5, 0.55], [0.45, 0.4, 0.45], 'r-')
                    
                    # Mouth
                    ax.plot([0.35, 0.65], [0.3, 0.3], 'r-')
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_title("Facial Landmarks")
                    ax.axis('off')
                    
                    st.pyplot(fig)
                
                with landmark_cols[1]:
                    st.markdown("""
                    **Key Facial Landmarks:**
                    
                    - 6 points for each eye
                    - 5 points for each eyebrow
                    - 9 points for the nose
                    - 20 points for the mouth
                    - 17 points for the face contour
                    
                    These landmarks provide precise spatial information about facial features and their movements.
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
    
    with tab4:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Real-time Prediction
            
            Once trained, the model can process live video for real-time gesture recognition:
            
            1. **Frame Capture**: Getting frames from video stream
            2. **Face Detection**: Locating face in each frame
            3. **Landmark Detection**: Identifying facial landmarks
            4. **Feature Extraction**: Processing landmarks for model input
            5. **Sequence Formation**: Collecting frames into temporal sequence
            6. **Prediction**: Running the CNN-LSTM model on the sequence
            7. **Output**: Displaying the recognized gesture
            
            The system operates with minimal latency to provide responsive feedback.
            """)
        
        with col2:
            st.image(FACIAL_RECOGNITION_IMAGES[3], use_column_width=True)
            st.caption("Real-time facial gesture recognition")
            
            # Add an interactive demo
            st.markdown("### Try a Demo Prediction")
            selected_gesture = st.selectbox(
                "Select a gesture to simulate prediction",
                ["Eye Blink", "Eyebrow Raise", "Head Nod (Yes)", "Head Shake (No)", "Neutral"]
            )
            
            if st.button("Run Prediction"):
                confidence = np.random.uniform(0.85, 0.98)
                st.success(f"Predicted Gesture: **{selected_gesture}** (Confidence: {confidence:.2f})")
                
                # Show a progress bar for processing
                progress_bar = st.progress(0)
                for i in range(100):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    # Add a slight delay
                    import time
                    time.sleep(0.01)

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
        
        st.image(AI_DATA_VIZ_IMAGES[2], use_column_width=True)
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
        
        st.image(AI_DATA_VIZ_IMAGES[3], use_column_width=True)
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
        st.image(PRESENTATION_SLIDE_IMAGES[0], use_column_width=True)
        st.caption("Interactive applications of facial recognition")
        
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
        
        st.image(PRESENTATION_SLIDE_IMAGES[1], use_column_width=True)
        st.caption("Presentation of gesture recognition applications")
    
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
