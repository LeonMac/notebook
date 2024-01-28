import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import rcParams
# from mpldatacursor import datacursor

###########################################
# Config CONSTANT 
NUM   = 50  # sample point number along X axis
DEN   = 20  # sample point number along Y axis
INTV  = 200 # animation display interval(ms)
R     = 1   # the unit circle radius 
YLIM  = R*3 # Y axis limit
PI    = np.pi
PD    = 1   # Period, must be INT

x_max = 2*PI*PD # float
X_MAX = int(x_max) + PD*6

# Config Chart
rcParams['figure.figsize'] = 16, 9 #X_MAX, 9

LW    = 4   # linewidth for chart plot
lw    = 2   # support line width

TAN_EN  = False # if display tan

# chat plt color config:
SIN_CLR = '#ff0000'
COS_CLR = '#00ff00'
TAN_CLR = '#0000ff'
SUP_CLR = '#dddddd' # grey
BLK_CLR = '#000000' # black
RAD_CLR = '#111111'

# Animation config
FPS = 10

# fig init
fig, ax = plt.subplots(1,1)
plt.axis('equal')
#polar = plt.subplot(111, projection='polar') # polar coodinator,not compatible

# text init
STXT = plt.text(0, 0, '', fontsize=12)  # sine annotation
CTXT = plt.text(0, 0, '', fontsize=12)  # cos  annotation

# annote init
ANNOT = ax.annotate('', xy = (0,0), xytext=(20,20), textcoords='offset points', 
                    bbox=dict(boxstyle='round', fc='w'), arrowprops=dict(arrowstyle="->"))
ANNOT.set_visible(False)

# set x/y limit
ax.set_xlim([0, x_max])
ax.set_ylim([-1*YLIM, 1*YLIM])

# X, Y range
X     = np.linspace(0, x_max, NUM)    # radian
# theta = np.arange(  0, 2*PI, 1/NUM)   # degree, not used

# triangle function
SY = R * np.sin(X)
CY = R * np.cos(X)
TY = R * np.tan(X)

# x axis path plot
ax.plot(3*X-3, 0*SY,    linewidth=1, color=SUP_CLR)  
# y axis path plot
ax.plot(0*X,   YLIM*SY, linewidth=1, color=SUP_CLR)

# unit circle path
ax.plot(CY, SY, linewidth=1, color = SUP_CLR)

X_shift = X + R # right shift radius for sin/cosin/tan plot

# sine path
SINcurve = ax.plot(X_shift, SY, linewidth=1, color = SIN_CLR) 

# cosin path
COScurve = ax.plot(X_shift, CY, linewidth=1, color = COS_CLR)

# interaction not work..
# ax.set_title('Click the position on curve')
# datacursor(SINcurve)
# datacursor(COScurve)


# tan path 
if TAN_EN:
    TANcurve = ax.plot(X_shift, TY, linewidth=1, color = TAN_CLR) 
#     datacursor(tancurve)

# fig, ax = plt.subplots()
# ax.scatter(np.random.rand(100), np.random.rand(100))

# # 编写回调函数
# def onclick(event):
#     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           ('double' if event.dblclick else 'single', event.button,
#            event.x, event.y, event.xdata, event.ydata))

# # 将回调函数连接到事件管理器上
# cid = fig.canvas.mpl_connect('button_press_event', onclick)

# plt.show()

# def fire_annot(chart, ind):
#     pos = chart.get_offset()[ind['ind'][0]]
#     ANNOT.xy = pos
#     msg = 'test'
#     ANNOT.set_tet(msg)
    
# def mosue_capture(evt):
#     vis = ANNOT.get_visible()
#     if evt.inaxes == ax:
#         cont, ind = SINcurve.contains(evt)
#         print (f'cont={cont}, ind={ind}')
#     else:
#         pass
#         # print (f'out of axes')
        
# fig.canvas.mpl_connect('motion_notify_event', mosue_capture)

# plt.show()

# https://juejin.cn/post/7031710412030623752
# class LineBuilder:
#     def __init__(self, line):
#         self.line = line
#         self.xs = list(line.get_xdata())
#         self.ys = list(line.get_ydata())
#         self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

#     def __call__(self, event):
#         print('click', event)
#         if event.inaxes!=self.line.axes: return
#         self.xs.append(event.xdata)
#         self.ys.append(event.ydata)
#         self.line.set_data(self.xs, self.ys)
#         self.line.figure.canvas.draw()

# fig, ax = plt.subplots()
# ax.set_title('click to build line segments')
# line, = ax.plot([0], [0])  # empty line
# linebuilder = LineBuilder(line)

# plt.show()
# https://www.bookstack.cn/read/Matplotlib/7.3.md
class DraggableRectangle:
    def __init__(self, rect):
        self.rect = rect
        self.press = None

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.rect.axes:
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        print('event contains', self.rect.xy)
        self.press = self.rect.xy, (event.xdata, event.ydata)

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.press is None or event.inaxes != self.rect.axes:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)
        self.rect.set_alpha(0.5)
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        """Clear button press information."""
        self.press = None
        self.rect.set_alpha(1.)
        self.rect.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

fig, ax = plt.subplots()
rects = ax.bar(range(10), 20*np.random.rand(10))
drs = []
for rect in rects:
    dr = DraggableRectangle(rect)
    dr.connect()
    drs.append(dr)

plt.show()
