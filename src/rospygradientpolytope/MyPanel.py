import rospy
from PyQt5.QtWidgets import QWidget, QPushButton
from PyQt5 import uic


class MyPanel(QWidget):

    def __init__(self):
        super(MyPanel, self).__init__()

        # Load the UI file
        ui_file = '/home/imr/catkin_ws_build/src/rospygradientpolytope/src/rospygradientpolytope/interactive_ik.ui'
        uic.loadUi(ui_file, self)

        # Find the QPushButton widget with objectName 'my_button'
        self.my_button = self.findChild(QPushButton, 'run_ik')

        # Connect the clicked signal of the QPushButton widget to the button_callback function
        self.my_button.clicked.connect(self.button_callback)

    def button_callback(self):
        # Do something when the button is clicked
        rospy.loginfo('Button clicked')




