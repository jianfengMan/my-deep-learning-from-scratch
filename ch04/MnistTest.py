import wx
import wx.xrc
import numpy as np


class MnistTestWindow(wx.Frame):

    def __init__(self, network, parent=None):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"MnistTest", pos=wx.DefaultPosition,
                          size=wx.Size(500, 400), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.m_statusBar1 = self.CreateStatusBar(1, wx.STB_SIZEGRIP, wx.ID_ANY)
        self.m_statusBar1.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.m_toolBar1 = self.CreateToolBar(wx.TB_HORIZONTAL, wx.ID_ANY)
        self.m_toolBar1.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.m_clearBtn = wx.Button(self.m_toolBar1, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_toolBar1.AddControl(self.m_clearBtn)
        self.m_predictBtn = wx.Button(self.m_toolBar1, wx.ID_ANY, u"Predict", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_toolBar1.AddControl(self.m_predictBtn)
        self.m_autoPredictCkb = wx.CheckBox(self.m_toolBar1, wx.ID_ANY, u"AutoPredict", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_toolBar1.AddControl(self.m_autoPredictCkb)
        self.m_toolBar1.Realize()

        bSizer1 = wx.BoxSizer(wx.HORIZONTAL)

        sbSizer1 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Draw"), wx.VERTICAL)

        self.m_drawPanel = wx.Panel(sbSizer1.GetStaticBox(), wx.ID_ANY, wx.Point(-1, -1), wx.Size(280, 280),
                                    wx.TAB_TRAVERSAL)
        self.m_drawPanel.SetForegroundColour(wx.Colour(0, 0, 0))
        self.m_drawPanel.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.m_drawPanel.SetMaxSize(wx.Size(280, 280))

        sbSizer1.Add(self.m_drawPanel, 1, wx.EXPAND | wx.ALL, 5)

        bSizer1.Add(sbSizer1, 1, wx.ALIGN_CENTER | wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 5)

        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Result"), wx.VERTICAL)

        bSizer23 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText13 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"0", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText13.Wrap(-1)

        bSizer23.Add(self.m_staticText13, 0, wx.ALL, 5)

        self.predictRateList = []
        self.m_gauge13 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge13.SetValue(0)
        self.predictRateList.append(self.m_gauge13)
        bSizer23.Add(self.m_gauge13, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer23, 1, wx.EXPAND, 5)

        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText1 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize,
                                           0)
        self.m_staticText1.Wrap(-1)

        bSizer2.Add(self.m_staticText1, 0, wx.ALL, 5)

        self.m_gauge1 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                 wx.GA_HORIZONTAL)
        self.m_gauge1.SetValue(0)
        self.predictRateList.append(self.m_gauge1)
        bSizer2.Add(self.m_gauge1, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer2, 1, wx.EXPAND, 5)

        bSizer21 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText11 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"2", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText11.Wrap(-1)

        bSizer21.Add(self.m_staticText11, 0, wx.ALL, 5)

        self.m_gauge11 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge11.SetValue(0)
        self.predictRateList.append(self.m_gauge11)
        bSizer21.Add(self.m_gauge11, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer21, 1, wx.EXPAND, 5)

        bSizer22 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText12 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"3", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText12.Wrap(-1)

        bSizer22.Add(self.m_staticText12, 0, wx.ALL, 5)

        self.m_gauge12 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge12.SetValue(0)
        self.predictRateList.append(self.m_gauge12)
        bSizer22.Add(self.m_gauge12, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer22, 1, wx.EXPAND, 5)

        bSizer24 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText14 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"4", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText14.Wrap(-1)

        bSizer24.Add(self.m_staticText14, 0, wx.ALL, 5)

        self.m_gauge14 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge14.SetValue(0)
        self.predictRateList.append(self.m_gauge14)
        bSizer24.Add(self.m_gauge14, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer24, 1, wx.EXPAND, 5)

        bSizer25 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText15 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"5", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText15.Wrap(-1)

        bSizer25.Add(self.m_staticText15, 0, wx.ALL, 5)

        self.m_gauge15 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge15.SetValue(0)
        self.predictRateList.append(self.m_gauge15)
        bSizer25.Add(self.m_gauge15, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer25, 1, wx.EXPAND, 5)

        bSizer26 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText16 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"6", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText16.Wrap(-1)

        bSizer26.Add(self.m_staticText16, 0, wx.ALL, 5)

        self.m_gauge16 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge16.SetValue(0)
        self.predictRateList.append(self.m_gauge16)
        bSizer26.Add(self.m_gauge16, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer26, 1, wx.EXPAND, 5)

        bSizer27 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText17 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"7", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText17.Wrap(-1)

        bSizer27.Add(self.m_staticText17, 0, wx.ALL, 5)

        self.m_gauge17 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge17.SetValue(0)
        self.predictRateList.append(self.m_gauge17)
        bSizer27.Add(self.m_gauge17, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer27, 1, wx.EXPAND, 5)

        bSizer28 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText18 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"8", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText18.Wrap(-1)

        bSizer28.Add(self.m_staticText18, 0, wx.ALL, 5)

        self.m_gauge18 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge18.SetValue(0)
        self.predictRateList.append(self.m_gauge18)
        bSizer28.Add(self.m_gauge18, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer28, 1, wx.EXPAND, 5)

        bSizer29 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText19 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"9", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText19.Wrap(-1)

        bSizer29.Add(self.m_staticText19, 0, wx.ALL, 5)

        self.m_gauge19 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge19.SetValue(0)
        self.predictRateList.append(self.m_gauge19)
        bSizer29.Add(self.m_gauge19, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer29, 1, wx.EXPAND, 5)

        bSizer1.Add(sbSizer2, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.m_clearBtn.Bind(wx.EVT_BUTTON, self.onClear)
        self.m_predictBtn.Bind(wx.EVT_BUTTON, self.onPredict)
        self.m_drawPanel.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.m_drawPanel.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.m_drawPanel.Bind(wx.EVT_MOTION, self.onMotion)

        self.m_drawPanel.Bind(wx.EVT_PAINT, self.onPaint)
        self.isDrawing = False
        self.network = network
        self.mnistArray = np.zeros(28 * 28)
        self.lastX, self.lastY = 0, 0

    def onClear(self, e):
        self.mnistArray = np.zeros(28 * 28)
        for g in self.predictRateList:
            g.SetValue(0)
        self.m_statusBar1.SetStatusText('Draw a single number above.')
        self.redraw()

    def onPredict(self, e):
        pd = self.network.predict(self.mnistArray)
        for i in range(len(pd)):
            self.predictRateList[i].SetValue(pd[i]*100)
        print(pd)
        self.m_statusBar1.SetStatusText('I guest the number is: %d' % np.argmax(pd))

    def onPaint(self, e):
        dc = wx.PaintDC(self.m_drawPanel)
        pen = wx.Pen(wx.Colour(100, 100, 100))
        brush = wx.Brush(wx.Colour(0, 0, 0))
        dc.SetPen(pen)
        dc.SetBrush(brush)
        self.drawGrid(dc)

    def drawGrid(self, dc):
        for p in range(0, 280, 10):
            dc.DrawLine(p, 0, p, 280)
            dc.DrawLine(0, p, 280, p)

        for x in range(28):
            for y in range(28):
                d = self.mnistArray[x + y * 28]
                if d != 0:
                    dc.DrawRectangle(x * 10 + 1, y * 10 + 1, 9, 9)

    def redraw(self):
        self.m_drawPanel.Refresh()

    def onClose(self, e):
        quit()

    def onLeftDown(self, event):
        self.isDrawing = True

    def onLeftUp(self, event):
        self.isDrawing = False
        if self.m_autoPredictCkb.IsChecked():
            self.onPredict(event)

    def onMotion(self, event):
        if self.isDrawing:
            pos = event.Position
            x = int(pos.x / 10)
            y = int(pos.y / 10)
            if x != self.lastX or y != self.lastY:
                self.lastX = x
                self.lastY = y
                self.mnistArray[x + y * 28] = 1
                self.redraw()

    def __del__(self):
        pass