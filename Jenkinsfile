#!groovy
CHAT_CHANNEL = '#deep_learning'
DEVELOPERS = ['mmoser@km3net.de']

def projectProperties = [
        disableConcurrentBuilds(),
        gitLabConnection('KM3NeT GitLab')
]
properties(projectProperties)

node('master') {

    // Start with a clean workspace
    // cleanWs()
    checkout scm

    def docker_image = docker.build("orcanet-container:${env.BUILD_ID}")

    docker_image.inside("-u root:root") {
        gitlabBuilds(builds: ["install", "test"]) {
            updateGitlabCommitStatus name: 'install', state: 'pending'
            stage('build') {
                try {
                    sh """
                        cd cnns
                        pip install .
                    """
                    updateGitlabCommitStatus name: 'install', state: 'success'
                } catch (e) {
                    // sendChatMessage("Build Failed")
                    sendMail("Build Failed")
                    updateGitlabCommitStatus name: 'install', state: 'failed'
                    throw e
                }
            }
            updateGitlabCommitStatus name: 'test', state: 'pending'
            stage('test') {
                try {
                    sh """
                        python -c "import cnns"
                    """
                    updateGitlabCommitStatus name: 'test', state: 'success'
                } catch (e) {
                    // sendChatMessage("Build Failed")
                    sendMail("Build Failed")
                    updateGitlabCommitStatus name: 'test', state: 'failed'
                    throw e
                }
            }
        }
    }
}


def sendChatMessage(message, channel=CHAT_CHANNEL) {
    rocketSend channel: channel, message: "${message} - [Build ${env.BUILD_NUMBER} ](${env.BUILD_URL})"
}


def sendMail(subject, message='', developers=DEVELOPERS) {
    for (int i = 0; i < developers.size(); i++) {
        def developer = DEVELOPERS[i]
        emailext (
            subject: "$subject - Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
            body: """
                <p>$message</p>
                <p>Check console output at <a href ='${env.BUILD_URL}'>${env.BUILD_URL}</a> to view the results.</p>
            """,
            to: developer
        )
    }
}

