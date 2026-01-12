pipeline {
    agent any

    environment {
        VENV_DIR = "venv"
    }
    stages {

        stage("Checkout Code") {
            steps {
                checkout scm
            }
        }

        stage('Check Python') {
            steps {
                bat 'python --version'
            }
        }
        stage('Check pip') {
            steps {
                bat '''
                python -m ensurepip --upgrade
                python -m pip install --upgrade pip setuptools wheel
                python -m pip --version
                '''
            }
        }
        stage('Install Dependencies') {
            steps {
                bat '''
                python -m pip install -r requirements.txt'''
            }
        }
        stage('cleanup folders started'){
            steps{
                bat '''
cd "%WORKSPACE%"
echo Cleaning old generated & report directories...

rmdir /s /q json_output || echo json_output not found
rmdir /s /q logs || echo logs not found
rmdir /s /q cleaned_data_output || echo cleaned_data_output not found
rmdir /s /q plots || echo plots not found
rmdir /s /q validation_reports || echo validation_reports not found
rmdir /s /q transformed_cleaned_data_output || echo transformed_cleaned_data_output not found
rmdir /s /q test-reports || echo test-reports not found
rmdir /s /q reports || echo reports not found
rmdir /s /q dashboard_reports || echo dashboard reports not found
rmdir /s /q htmlcov || echo coverage report not found

mkdir json_output
mkdir cleaned_data_output
mkdir plots
mkdir validation_reports
mkdir reports
mkdir logs
mkdir dashboard_reports
mkdir transformed_cleaned_data_output

echo Cleanup complete!
'''
            }
        }

        stage('Run Data Cleaning Process') {
            steps {
                bat'''
                python run_reports.py
                '''
            }
        }
        stage('Run generate transformation') {
            steps {
                bat'''
                python generate_transformation.py'''
            }
        }
        stage('Run apply transformation') {
            steps {
                bat'''
                python apply_transformation.py'''
            }
        }
        stage('Upload plots to GitHub Release') {
    steps {
        withCredentials([string(credentialsId: 'GITHUB_TOKEN', variable: 'GITHUB_TOKEN')]) {
            bat '''
            python cleanup_upload_images.py
            '''
        }
    }
}
        stage('Run generate metrics pdf ') {
            steps {
                bat'''
                python dashboard_pdf.py'''
            }
        }
    }
    post {
    success {
        emailext(
            to: "sahithi251999@gmail.com",
            subject: "Dashboard Reports - Jenkins Build #${BUILD_NUMBER}",
            body: """
Hello,

The Jenkins pipeline has completed successfully.

Please find the dashboard reports attached.

Job: ${JOB_NAME}
Build: #${BUILD_NUMBER}

Regards,
Jenkins
""",
            attachmentsPattern: "dashboard_reports/**/*.*"
        )
    }

    failure {
        emailext(
            to: "sahithi251999@gmail.com",
            subject: "Jenkins Build Failed - ${JOB_NAME} #${BUILD_NUMBER}",
            body: "The pipeline failed. Please check Jenkins logs."
        )
    }
}


}
