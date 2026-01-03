pipeline {
    agent any

    stages {
        stage('Clone Git Repo') {
            steps {
                echo 'Git repo cloned automatically by SCM'
            }
        }

        stage('Check Python') {
            steps {
                bat 'python --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install -r requirements.txt'
            }
        }

        stage('Run Data Cleaning Process') {
            steps {
                bat 'python run_reports.py'
            }
        }
        stage('Run generate transformation') {
            steps {
                bat 'python generate_transformation.py'
            }
        }
        stage('Run apply transformation') {
            steps {
                bat 'python apply_transformation.py'
            }
        }
        stage('Run Streamlit Dashboard Creation') {
            steps {
                bat 'streamlit run src/etl_dashboard.py'
            }
        }
    }
}
