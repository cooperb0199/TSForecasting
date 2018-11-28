#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:56:41 2018

@author: bencooper
"""

class RawTwitterObject:
    
    def __init__(self, TwitterId, Nickname, MessageId, Message, TimeStamp):
        self.TwitterId = TwitterId
        self.Nickname = Nickname
        self.MessageId = MessageId
        self.Message = Message
        self.TimeStamp = TimeStamp