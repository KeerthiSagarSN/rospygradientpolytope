#!/usr/bin/env python

import rospy
import os
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout
from qt_gui.plugin import Plugin

# Import the UI module generated from your_panel.ui
from rospygradientpolytope.interactive_ik import Ui_Dialog

class MyPanel(QWidget):
    def __init__(self, context):
        super(MyPanel, self).__init__()
        self.setObjectName('MyPanel')
        uic.loadUi('/home/imr/catkin_ws_build/src/rospygradientpolytope/src/rospygradientpolytope/interactive_ik.ui', self)

        '''
        # Create buttons
        button1 = QPushButton('Button 1')
        button2 = QPushButton('Button 2')

        # Connect buttons to their respective slots
        button1.clicked.connect(self.on_button1_clicked)
        button2.clicked.connect(self.on_button2_clicked)

        # Create layouts and add buttons to them
        hlayout = QHBoxLayout()
        hlayout.addWidget(button1)
        hlayout.addWidget(button2)

        vlayout = QVBoxLayout()
        vlayout.addLayout(hlayout)

        # Set the layout for the widget
        self.setLayout(vlayout)
        '''

    def on_button1_clicked(self):
        print('Button 1 clicked')

    def on_button2_clicked(self):
        print('Button 2 clicked')

class MyPlugin(Plugin):
    def __init__(self, context):
        super(MyPlugin, self).__init__(context)
        self.setObjectName('MyPlugin')

        # Create the widget
        self._widget = MyPanel(context)

        # Add the widget to the user interface
        context.add_widget(self._widget)

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('my_rviz_panel')
    import sys

    # Create an instance of QApplication
    app = QApplication([])

    # Create a widget
    widget = QWidget()

    # Show the widget
    widget.show()
    # Create the plugin and start the Qt event loop
    plugin = MyPlugin(None)
    plugin._widget.show()
    # Run the event loop
    sys.exit(app.exec_())

    # import sys
    # if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtWidgets.QApplication.instance().exec_()
