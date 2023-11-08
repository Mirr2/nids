from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class mplwidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        # 두 개의 서브플롯 생성
        self.axes_loss = fig.add_subplot(211)  # 손실을 위한 축
        self.axes_acc = fig.add_subplot(212)  # 정확도를 위한 축

        # 각 축에 대한 초기화 설정
        self.lines_loss, = self.axes_loss.plot([], [], label='Loss')
        self.lines_acc, = self.axes_acc.plot([], [], label='Accuracy')

        # 범례 설정
        self.axes_loss.legend(loc='upper right')
        self.axes_acc.legend(loc='upper right')

        super(mplwidget, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_graph(self, loss_data, acc_data):
        # 그래프 데이터 업데이트

        # 손실 데이터 업데이트
        xdata_loss = range(len(loss_data))
        ydata_loss = loss_data
        self.lines_loss.set_xdata(xdata_loss)
        self.lines_loss.set_ydata(ydata_loss)
        self.axes_loss.relim()
        self.axes_loss.autoscale_view()

        # 정확도 데이터 업데이트
        xdata_acc = range(len(acc_data))
        ydata_acc = acc_data
        self.lines_acc.set_xdata(xdata_acc)
        self.lines_acc.set_ydata(ydata_acc)
        self.axes_acc.relim()
        self.axes_acc.autoscale_view()

        # 그래프 다시 그리기
        self.draw()