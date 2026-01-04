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
                bat '''
                start /B python -m streamlit run src/etl_dashboard.py --server.port=8501 --server.address=127.0.0.1
                timeout /t 3
                echo http://localhost:8501
                '''

            }
        }
    }
}
