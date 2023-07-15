import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class MailTest(object):
    def __init__(self, username, authKey, recv, title, content,
                 email_host='smtp.qq.com', port=587):
        self.username = username
        self.authKey = authKey
        self.recv = recv
        self.title = title
        self.content = content
        self.email_host = email_host
        self.port = port

    def send_mail(self):
        msg = MIMEMultipart()

        # 发送内容的对象
        msg.attach(MIMEText(self.content))  # 邮件正文的内容
        msg['Subject'] = self.title  # 邮件主题
        msg['From'] = self.username  # 发送者账号
        msg['To'] = self.recv  # 接收者账号列表
        self.smtp = smtplib.SMTP(self.email_host, self.port)
        # 发送邮件服务器的对象
        self.smtp.login(self.username, self.authKey)
        try:
            self.smtp.sendmail(self.username, self.recv, msg.as_string())
        except Exception as e:
            print('出错了', e)
        else:
            print('发送成功！')


def __del__(self):
    self.smtp.quit()


# 调用封装
if __name__ == '__main__':
    m = MailTest(username='***@qq.com', authKey='***', recv='***@qq.com',
                 title='hello gugu', content='python 发个邮件')
    m.send_mail()
