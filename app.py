import streamlit as st
import pickle
import fitz
import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="AI Skill Gap Detector",
    page_icon="🎯",
    layout="wide"
)

# Load Models
@st.cache_resource
def load_models():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, tfidf, le

model, tfidf, le = load_models()

# Skills Database
ROLE_SKILLS = {
    "INFORMATION-TECHNOLOGY": ["python","java","sql","linux","networking","cloud","aws","docker","kubernetes","cybersecurity"],
    "DATA-SCIENCE": ["python","machine learning","deep learning","pandas","numpy","tensorflow","pytorch","sql","statistics","tableau"],
    "BUSINESS-DEVELOPMENT": ["sales","crm","negotiation","market research","communication","excel","powerpoint","linkedin","b2b","strategy"],
    "ADVOCATE": ["legal research","litigation","drafting","communication","negotiation","case management","client counseling","legal writing"],
    "CHEF": ["cooking","menu planning","food safety","kitchen management","inventory","nutrition","baking","food presentation"],
    "ENGINEERING": ["autocad","solidworks","matlab","project management","problem solving","technical drawing","quality control"],
    "ACCOUNTANT": ["accounting","tally","excel","taxation","auditing","financial reporting","gst","balance sheet","tds"],
    "FINANCE": ["financial analysis","excel","valuation","investment","risk management","portfolio","bloomberg","python","sql"],
    "FITNESS": ["personal training","nutrition","workout planning","anatomy","injury prevention","client management","cpr"],
    "AVIATION": ["flight operations","navigation","safety procedures","communication","technical knowledge","teamwork"],
    "SALES": ["communication","crm","negotiation","target achievement","product knowledge","excel","cold calling","b2b"],
    "BANKING": ["banking operations","kyc","loan processing","excel","financial products","customer service","risk assessment"],
    "HEALTHCARE": ["patient care","clinical skills","medical knowledge","communication","empathy","record keeping","diagnosis"],
    "CONSULTANT": ["problem solving","communication","data analysis","presentation","excel","powerpoint","research","strategy"],
    "CONSTRUCTION": ["project management","autocad","site management","budgeting","safety","planning","quality control"],
    "PUBLIC-RELATIONS": ["communication","media relations","content writing","social media","crisis management","press release"],
    "HR": ["recruitment","payroll","employee relations","hrms","communication","training","performance management","excel"],
    "DESIGNER": ["photoshop","illustrator","figma","ui/ux","creativity","typography","color theory","adobe xd"],
    "ARTS": ["creativity","drawing","painting","art history","design","communication","portfolio management"],
    "TEACHER": ["communication","lesson planning","classroom management","subject expertise","patience","assessment","e-learning"],
    "APPAREL": ["fashion design","pattern making","sewing","trend analysis","fabric knowledge","cad","retail management"],
    "DIGITAL-MEDIA": ["social media","content creation","seo","google analytics","photoshop","video editing","copywriting"],
    "AGRICULTURE": ["crop management","soil science","irrigation","pesticides","farming techniques","agri business"],
    "AUTOMOBILE": ["mechanical knowledge","diagnostics","autocad","engine repair","electrical systems","quality control"],
    "BPO": ["communication","customer service","data entry","crm","problem solving","multitasking","typing speed"]
}

# Specific Job Titles
SPECIFIC_TITLES = {
    "INFORMATION-TECHNOLOGY": {
        "docker":           "DevOps Engineer",
        "kubernetes":       "DevOps Engineer",
        "aws":              "Cloud Engineer",
        "cloud":            "Cloud Engineer",
        "cybersecurity":    "Cybersecurity Engineer",
        "java":             "Software Development Engineer",
        "python":           "Software Development Engineer",
        "linux":            "System Administrator",
        "networking":       "Network Engineer",
        "machine learning": "AI Engineer",
        "deep learning":    "AI Engineer",
        "tensorflow":       "AI Engineer",
        "pytorch":          "AI Engineer",
        "sql":              "Database Engineer"
    },
    "DATA-SCIENCE": {
        "machine learning": "AI Engineer",
        "deep learning":    "AI Engineer",
        "tensorflow":       "AI Engineer",
        "pytorch":          "AI Engineer",
        "tableau":          "Data Analyst",
        "statistics":       "Data Scientist",
        "pandas":           "Data Scientist",
        "numpy":            "Data Scientist",
        "sql":              "Data Analyst",
        "python":           "Software Development Engineer"
    },
    "DESIGNER": {
        "figma":      "UI/UX Designer",
        "ui/ux":      "UI/UX Designer",
        "photoshop":  "Graphic Designer",
        "illustrator":"Graphic Designer",
        "adobe xd":   "UI/UX Designer"
    },
    "DIGITAL-MEDIA": {
        "seo":              "SEO Specialist",
        "google analytics": "Digital Marketing Analyst",
        "content creation": "Content Creator",
        "video editing":    "Video Editor",
        "social media":     "Social Media Manager"
    },
    "FINANCE": {
        "valuation":      "Investment Analyst",
        "portfolio":      "Portfolio Manager",
        "risk management":"Risk Analyst",
        "investment":     "Investment Banker",
        "bloomberg":      "Financial Analyst"
    },
    "HR": {
        "recruitment": "Talent Acquisition Specialist",
        "payroll":     "Payroll Manager",
        "training":    "L&D Specialist",
        "hrms":        "HR Manager"
    },
    "ENGINEERING": {
        "autocad":    "CAD Engineer",
        "solidworks": "Mechanical Engineer",
        "matlab":     "Simulation Engineer"
    },
    "ACCOUNTANT": {
        "auditing": "Auditor",
        "gst":      "Tax Consultant",
        "tally":    "Accountant"
    },
    "BANKING": {
        "kyc":           "Compliance Officer",
        "loan processing":"Loan Officer",
        "risk assessment":"Risk Analyst"
    },
    "HEALTHCARE": {
        "diagnosis":      "Clinical Doctor",
        "patient care":   "Healthcare Professional",
        "clinical skills":"Clinical Specialist"
    },
    "CONSULTANT": {
        "data analysis":  "Business Analyst",
        "strategy":       "Strategy Consultant",
        "research":       "Research Analyst"
    }
}

# Learning Resources
LEARNING_RESOURCES = {
    "python":           "https://www.coursera.org/learn/python",
    "machine learning": "https://www.coursera.org/learn/machine-learning",
    "deep learning":    "https://www.coursera.org/specializations/deep-learning",
    "sql":              "https://www.w3schools.com/sql/",
    "aws":              "https://aws.amazon.com/training/",
    "docker":           "https://docs.docker.com/get-started/",
    "kubernetes":       "https://kubernetes.io/docs/tutorials/",
    "pandas":           "https://pandas.pydata.org/docs/",
    "tensorflow":       "https://www.tensorflow.org/tutorials",
    "tableau":          "https://www.tableau.com/learn/training",
    "excel":            "https://www.coursera.org/learn/excel-essentials",
    "figma":            "https://www.figma.com/resources/learn-design/",
    "photoshop":        "https://www.adobe.com/learn/photoshop",
    "seo":              "https://moz.com/learn/seo",
    "communication":    "https://www.coursera.org/learn/communication-skills",
    "java":             "https://www.coursera.org/learn/java-programming",
    "linux":            "https://www.edx.org/learn/linux",
    "networking":       "https://www.coursera.org/learn/computer-networking",
    "cloud":            "https://www.coursera.org/learn/cloud-computing",
    "cybersecurity":    "https://www.coursera.org/learn/cybersecurity-fundamentals",
    "pytorch":          "https://pytorch.org/tutorials/",
    "statistics":       "https://www.coursera.org/learn/statistics-for-data-science",
    "autocad":          "https://www.autodesk.com/certification/learn",
    "powerpoint":       "https://www.coursera.org/learn/powerpoint",
    "negotiation":      "https://www.coursera.org/learn/negotiation",
    "social media":     "https://www.coursera.org/learn/social-media-marketing",
    "video editing":    "https://www.coursera.org/learn/video-editing",
    "content creation": "https://www.coursera.org/learn/content-marketing",
    "numpy":            "https://numpy.org/learn/",
    "kubernetes":       "https://kubernetes.io/docs/tutorials/",
    "docker":           "https://docs.docker.com/get-started/",
    "b2b":              "https://www.coursera.org/learn/sales-training",
    "crm":              "https://www.coursera.org/learn/crm",
    "recruitment":      "https://www.coursera.org/learn/human-resources",
    "photoshop":        "https://www.adobe.com/learn/photoshop",
    "illustrator":      "https://www.adobe.com/learn/illustrator",
}

# Helper Functions
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def predict_role(cleaned_text):
    vectorized = tfidf.transform([cleaned_text])
    prediction = model.predict(vectorized)
    role = le.inverse_transform(prediction)[0]
    try:
        proba    = model.predict_proba(vectorized)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3     = [(le.inverse_transform([i])[0], round(proba[i]*100, 2)) for i in top3_idx]
    except:
        top3 = [(role, 100.0)]
    return role, top3

def extract_skills(text):
    text_lower = text.lower()
    all_skills = set()
    for skills in ROLE_SKILLS.values():
        all_skills.update(skills)
    return [skill for skill in all_skills if skill in text_lower]

def get_skill_gap(found_skills, predicted_role):
    required = ROLE_SKILLS.get(predicted_role, [])
    missing  = [s for s in required if s not in found_skills]
    present  = [s for s in required if s in found_skills]
    return required, present, missing

def get_specific_title(role, found_skills):
    role_map = SPECIFIC_TITLES.get(role, {})
    for skill in found_skills:
        if skill in role_map:
            return role_map[skill]
    return role.replace("-", " ").title()

# UI Header
st.markdown("""
    <h1 style='text-align:center; color:#4A90E2;'>
        🎯 AI Skill Gap Detector
    </h1>
    <p style='text-align:center; color:gray; font-size:18px;'>
        Upload Resume → Predict Job Role → Detect Skill Gaps → Get Learning Roadmap
    </p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/resume.png", width=150)
    st.markdown("### 📊 About")
    st.info("""
**AI Skill Gap Detector**
- 🤖 6 ML Models trained
- 📄 2484 Resumes dataset
- 🏷️ 24 Job Categories
- 🎯 LightGBM Best Model
    """)
    st.markdown("---")
    st.markdown("### 📌 How to Use")
    st.markdown("""
1. Upload your PDF resume
2. View predicted job role
3. Check skill gap analysis
4. Follow learning roadmap
    """)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📄 Resume Analyzer",
    "📊 Skill Gap",
    "🗺️ Learning Roadmap"
])

# Tab 1: Resume Analyzer
with tab1:
    st.markdown("### 📤 Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Upload PDF Resume",
        type=["pdf"],
        help="Upload your resume in PDF format"
    )

    if uploaded_file:
        with st.spinner("🔍 Analyzing your resume..."):
            raw_text     = extract_text_from_pdf(uploaded_file)
            cleaned      = clean_text(raw_text)
            role, top3   = predict_role(cleaned)
            found_skills = extract_skills(cleaned)
            required, present, missing = get_skill_gap(found_skills, role)
            specific_title = get_specific_title(role, found_skills)

            st.session_state['role']          = role
            st.session_state['top3']          = top3
            st.session_state['found']         = found_skills
            st.session_state['required']      = required
            st.session_state['present']       = present
            st.session_state['missing']       = missing
            st.session_state['text']          = raw_text
            st.session_state['specific_title']= specific_title

        st.success("✅ Resume analyzed successfully!")

        # Specific Job Title Box
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1e3a5f,#2d6a9f);
                    padding:20px; border-radius:12px; margin:10px 0;
                    border-left:5px solid #4A90E2;'>
            <p style='color:#add8e6; margin:0; font-size:14px;'>💼 Specific Job Title</p>
            <h2 style='color:white; margin:5px 0;'>{specific_title}</h2>
            <p style='color:#add8e6; margin:0; font-size:13px;'>Based on your skills profile</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Predicted Category", role)
        with col2:
            st.metric("✅ Skills Found", len(found_skills))
        with col3:
            st.metric("❌ Skills Missing", len(missing))

        st.markdown("### 🏆 Top 3 Role Predictions")
        for r, prob in top3:
            st.progress(int(prob), text=f"{r} → {prob}%")

        st.markdown("### 📄 Resume Text Preview")
        with st.expander("Click to view extracted text"):
            st.text(raw_text[:2000])

# Tab 2: Skill Gap
with tab2:
    if 'role' not in st.session_state:
        st.warning("⚠️ Please upload your resume in Tab 1 first!")
    else:
        role     = st.session_state['role']
        present  = st.session_state['present']
        missing  = st.session_state['missing']
        required = st.session_state['required']

        st.markdown(f"### 🎯 Skill Gap Analysis for: `{role}`")
        score = int((len(present) / len(required)) * 100) if required else 0

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = score,
                title = {'text': "Profile Match %"},
                gauge = {
                    'axis'  : {'range': [0, 100]},
                    'bar'   : {'color': "#4A90E2"},
                    'steps' : [
                        {'range': [0,  40],  'color': "#ff4444"},
                        {'range': [40, 70],  'color': "#ffaa00"},
                        {'range': [70, 100], 'color': "#00cc44"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.pie(
                values=[len(present), len(missing)],
                names=['Skills Present', 'Skills Missing'],
                color_discrete_sequence=['#00cc44', '#ff4444'],
                title='Skills Overview'
            )
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### ✅ Skills You Have")
            for s in present:
                st.success(f"✅ {s.title()}")
            if not present:
                st.info("No matching skills found")
        with col4:
            st.markdown("### ❌ Skills You're Missing")
            for s in missing:
                st.error(f"❌ {s.title()}")
            if not missing:
                st.success("🎉 You have all required skills!")

# Tab 3: Learning Roadmap
with tab3:
    if 'missing' not in st.session_state:
        st.warning("⚠️ Please upload your resume in Tab 1 first!")
    else:
        missing        = st.session_state['missing']
        role           = st.session_state['role']
        specific_title = st.session_state.get('specific_title', role)

        st.markdown(f"### 🗺️ Personalized Learning Roadmap")
        st.markdown(f"**Target Role:** {specific_title}")

        if not missing:
            st.balloons()
            st.success("🎉 You have all required skills!")
        else:
            st.markdown(f"#### 📚 You need to learn **{len(missing)} skills**")

            for i, skill in enumerate(missing, 1):
                with st.expander(f"📌 Step {i}: Learn {skill.title()}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Skill:** {skill.title()}")
                        if i <= 3:
                            priority = "🔴 High Priority"
                            time_est = "3-4 weeks"
                        elif i <= 6:
                            priority = "🟡 Medium Priority"
                            time_est = "2-3 weeks"
                        else:
                            priority = "🟢 Low Priority"
                            time_est = "1-2 weeks"
                        st.markdown(f"**Priority:** {priority}")
                        st.markdown(f"**Est. Time:** {time_est}")
                    with col2:
                        link = LEARNING_RESOURCES.get(
                            skill.lower(),
                            f"https://www.coursera.org/search?query={skill.replace(' ', '+')}"
                        )
                        st.link_button("📖 Start Learning", link)

            st.markdown("### 📅 Learning Timeline")
            skills_chart = missing[:8]
            weeks_chart  = [4 if i < 3 else 2 if i < 6 else 1
                            for i in range(len(skills_chart))]
            fig = px.bar(
                x=skills_chart,
                y=weeks_chart,
                labels={'x': 'Skill', 'y': 'Weeks'},
                title='Estimated Learning Time per Skill',
                color=skills_chart,
            )
            st.plotly_chart(fig, use_container_width=True)