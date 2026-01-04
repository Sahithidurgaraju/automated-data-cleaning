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
        stage('cleanup folders started'){
            steps{
                bat '''
cd "%WORKSPACE%"
echo Cleaning old generated & report directories...

rmdir /s /q json_output || echo json_output not found
rmdir /s /q cleaned_data_output || echo cleaned_data_output not found
rmdir /s /q plots || echo plots not found
rmdir /s /q validation_reports || echo validation_reports not found
rmdir /s /q test-reports || echo test-reports not found
rmdir /s /q reports || echo reports not found
rmdir /s /q htmlcov || echo coverage report not found

mkdir json_output
mkdir cleaned_data_output
mkdir plots
mkdir validation_reports
mkdir reports

echo Cleanup complete!
'''
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
cd "%WORKSPACE%"
taskkill /IM streamlit.exe /F
start cmd /k "python -m streamlit run src/etldashboard.py --server.port=8501 --server.address=127.0.0.1"
echo http://localhost:8501
'''

            }
        }
    }
}
