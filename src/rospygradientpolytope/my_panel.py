#!/usr/bin/env python

import rospy
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from rviz import Panel

# Import the UI module generated from your_panel.ui
from rospygradientpolytope.interactive_ik import Ui_Dialog

class YourPanel(Panel):

    def __init__(self, parent=None):
        super(YourPanel, self).__init__(parent)

        # Create an instance of the UI from the module
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Connect signals and slots as necessary
        my_button = self.findChild(QPushButton, "run_ik")
        if my_button is not None:
            my_button.clicked.connect(self.on_my_button_clicked)

    def on_my_button_clicked(self):
        rospy.loginfo("My button clicked!")

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('your_panel')

    # Create an instance of your panel and add it to RViz
    panel = YourPanel()
    panel.setWindowTitle("Inverse kinematics- Capacity Margin")
    panel.setVisible(True)

    # Spin the ROS node to keep the panel alive
    rospy.spin()
