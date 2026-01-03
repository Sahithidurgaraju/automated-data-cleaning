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

        stage('Run ETL Code') {
            steps {
                bat 'python run_reports.py'
            }
        }
    }
}
