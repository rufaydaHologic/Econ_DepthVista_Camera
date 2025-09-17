import cv2
import ctypes
from ctypes import*
from input import get_integer
import threading
import numpy as np
import time
from enum import Enum
from datetime import datetime
import sys

depth_min = 1000
depth_max = 6000
depth_range = (depth_max-depth_min)*0.20
calibParamObtained = False
rgbdMappingflag = False
captureDone = False

#Preview Depth and IR Resolution 640x480
PRE_DEPTH_WIDTH = PRE_IR_WIDTH = 640
PRE_DEPTH_HEIGHT = PRE_IR_HEIGHT = 480

#Preview RGB HD Resolution 1280x720
PRE_RGB_HD_WIDTH = 1280
PRE_RGB_HD_HEIGHT = 720

'''
    Defining DeviceType Enum
'''
class DeviceType(Enum):
    RGBD_IRD = 0
    USB_IRD = 1
    GMSL_USB = 2
    PCIE_IRD = 3

'''
    Defining tofFrame Structure using ctypes
'''
class tofFrame(ctypes.Structure):
    _fields_ = [ ("frame_data",ctypes.POINTER(ctypes.c_uint8)),
                 ("width",ctypes.c_uint16),
                 ("height",ctypes.c_uint16),
                 ("pixel_format",ctypes.c_uint8),
                 ("size",ctypes.c_uint32),
                 ("time_stamp", ctypes.c_uint64)
               ]
'''
    Defining Frames Structure using ctypes
'''
class frames(ctypes.Structure):
    _fields_ = [ ("rgb", tofFrame),
                 ("ir", tofFrame),
                 ("raw_ir", tofFrame),
                 ("raw_depth", tofFrame),
                 ("depth_colormap", tofFrame),
                 ("confidence_frame", tofFrame),
                 ("IRA0RawFrame", tofFrame),
                 ("IRA1RawFrame", tofFrame),
                 ("IRA2RawFrame", tofFrame),
                 ("IRA0RawFrame_save", tofFrame), 
                 ("IRA1RawFrame_save", tofFrame),
                 ("IRA2RawFrame_save", tofFrame),
               ] 
'''
    Defining DeviceHandle Structure using ctypes
'''
class DeviceHandle(ctypes.Structure):
    _fields_ = [ ("serialNo", ctypes.c_char * 50)]

'''
    Defining DeviceInfo Structure using ctypes
'''
class DeviceInfo(ctypes.Structure):
    _fields_ = [  ("deviceName",ctypes.c_char * 50),
                  ("vid",ctypes.c_char * 5),
                  ("pid",ctypes.c_char * 5),
                  ("devicePath",ctypes.c_char * 500),
                  ("serialNo",ctypes.c_char * 50),
                  ("devType", ctypes.c_uint32)
               ]

'''
    Defining DepthPtr Structure using ctypes
'''
class DepthPtr(ctypes.Structure):
    _fields_ = [  ("X", ctypes.c_int),
                  ("Y", ctypes.c_int)
               ]

'''
    Defining IntrinsicCalibParams Structure using ctypes
'''
class IntrinsicCalibParams(ctypes.Structure):
    _fields_ = [  ("fx", ctypes.c_double),
                  ("fy", ctypes.c_double),
                  ("cx", ctypes.c_double),
                  ("cy", ctypes.c_double),
                  ("k1", ctypes.c_double),
                  ("k2", ctypes.c_double),
                  ("p1", ctypes.c_double),
                  ("p2", ctypes.c_double),
                  ("k3", ctypes.c_double)
               ]
    
'''
    Defining ExtrinsicCalibParams Structure using ctypes
'''
class ExtrinsicCalibParams(ctypes.Structure):
    _fields_ = [  ("rotationalVec", ctypes.c_double * 3 * 3),
                  ("translationalVec",ctypes.c_double * 3)
               ]
    
'''
    Defining CalibrationParams Structure using ctypes
'''
class CalibrationParams(ctypes.Structure):
    _fields_ =  [  ("depthCamVGAIntrinsic", IntrinsicCalibParams),
                   ("rgbCamVGAIntrinsic", IntrinsicCalibParams),
                   ("depthCamHDIntrinsic", IntrinsicCalibParams),
                   ("rgbCamHDIntrinsic", IntrinsicCalibParams),
                   ("VGAextrinsic", ExtrinsicCalibParams),
                   ("HDextrinsic", ExtrinsicCalibParams)
                ]

'''
    Defining Datamode Enum
'''
class DataMode(Enum):
    Depth_IR_Mode = 0
    Depth_Mode = 2
    IR_Mode = 3
    Depth_IR_RGB_VGA_Mode = 4
    Depth_IR_RGB_HD_Mode = 5
    RGB_VGA_Mode = 6
    RGB_HD_Mode = 7
    RGB_Full_HD_Mode = 8
    RGB_1200p_Mode = 9

'''
    Defining DepthRange Enum
'''
class DepthRange(Enum):
    NearRange = 0
    FarRange = 1

'''
    Defining AvgRegion Enum
'''
class AvgRegion(Enum):
    Center = 0
    CustomPtr = 1
    Exit = -1

if(sys.platform == "linux"):
    sharedLibrary = "libDepthVistaSDK.so"
elif(sys.platform == "win32"):
    sharedLibrary = "DepthVistaSDK.dll"
depthVistaLib = ctypes.CDLL(sharedLibrary)

'''
    Main Class: The initiator class. The program starts from here.
'''
class MainClass:

    initializeresult = depthVistaLib.Initialize
    initializeresult.restype = ctypes.c_int

    deviceCountResult = depthVistaLib.GetDeviceCount
    deviceCountResult.restype = ctypes.c_int
    deviceCountResult.argtypes = [ ctypes.POINTER(ctypes.c_uint32) ]

    getDeviceInfoResult = depthVistaLib.GetDeviceInfo
    getDeviceInfoResult.restype = ctypes.c_int
    getDeviceInfoResult.argtypes = [ctypes.c_uint32, ctypes.POINTER(DeviceInfo)]

    getDeviceListInfoResult = depthVistaLib.GetDeviceListInfo
    getDeviceListInfoResult.restype = ctypes.c_int
    getDeviceListInfoResult.argtypes = [ctypes.c_uint32, ctypes.POINTER(DeviceInfo)]
    
    openDeviceResult =  depthVistaLib.OpenDevice
    openDeviceResult.restype = ctypes.c_int
    openDeviceResult.argtypes = [DeviceInfo, ctypes.POINTER(DeviceHandle)]

    isOpenResult = depthVistaLib.IsOpened
    isOpenResult.restype = ctypes.c_int
    isOpenResult.argtypes = [ DeviceHandle ]

    closeDeviceResult = depthVistaLib.CloseDevice
    closeDeviceResult.restype = ctypes.c_int
    closeDeviceResult.argtypes = [ DeviceHandle ]

    deinitializeResult = depthVistaLib.DeInitialize
    deinitializeResult.restype = ctypes.c_int

    setdataModeResult = depthVistaLib.SetDataMode
    setdataModeResult.restype = ctypes.c_int
    setdataModeResult.argtypes = [DeviceHandle, ctypes.c_int32]

    setDepthRangeResult = depthVistaLib.SetDepthRange
    setDepthRangeResult.restype = ctypes.c_int
    setDepthRangeResult.argtypes = [DeviceHandle, ctypes.c_uint16]

    nextFrameResult = depthVistaLib.GetNextFrame
    nextFrameResult.restype = ctypes.c_int
    nextFrameResult.argtypes = [ DeviceHandle ]

    getFramesResult = depthVistaLib.GetFrames
    getFramesResult.restype = ctypes.c_int
    getFramesResult.argtypes = [DeviceHandle, ctypes.POINTER(frames)]
    
    updateColorMapResult = depthVistaLib.UpdateColorMap
    updateColorMapResult.restype = ctypes.c_int
    updateColorMapResult.argtypes = [DeviceHandle, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

    setRGBDMappingResult = depthVistaLib.SetRGBDMapping
    setRGBDMappingResult.restype = ctypes.c_int
    setRGBDMappingResult.argtypes = [DeviceHandle, ctypes.c_uint16]

    setAvgRegionResult = depthVistaLib.SetAvgRegion
    setAvgRegionResult.restype = ctypes.c_int
    setAvgRegionResult.argtypes = [DeviceHandle, ctypes.c_uint16]

    getSDKVersionResult = depthVistaLib.GetSDKVersion
    getSDKVersionResult.restype = ctypes.c_int
    getSDKVersionResult.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16)]

    setDepthPosResult = depthVistaLib.SetDepthPos
    setDepthPosResult.restype = ctypes.c_int
    setDepthPosResult.argtypes = [DeviceHandle, DepthPtr]

    getDevCalibParamsResult = depthVistaLib.GetDeviceCalibrationParams
    getDevCalibParamsResult.restype = ctypes.c_int
    getDevCalibParamsResult.argtypes = [DeviceHandle, ctypes.POINTER(CalibrationParams)]

    Thread_end = threading.Event()
    modechange = threading.Event()
    allFrames = frames()
    GetNextFrameThread = threading.Thread()
    thread_lock_flag = False
    thread_lock = threading.Lock()
    deviceHandle = DeviceHandle()
    datamode = DataMode.Depth_IR_RGB_VGA_Mode.value
    depthrange = DepthRange.FarRange.value
    depth_cap = False
    rgb_cap = False
    deviceSelected = False
    depthStreamStarted = False
    save_possible = False
    deviceCalibParams = CalibrationParams()

    '''
        METHOD NAME : __init__
        DESCRIPTION : First method to initiate all process
    '''
    def __init__(self):
        self.vid = None
        self.pid = None
        self.device_path = None
        self.device_name = None
        self.No_of_devices = None
        self.serialNo = None
        sdkMajorVer = ctypes.c_uint8()
        sdkMinorVer1 = ctypes.c_uint8()
        sdkMinorVer2 = ctypes.c_uint16()

        #Initializing all the APIs in shared library
        if(self.initializeresult() == 0):
            print("\nFailed to initialize device")
            exit(0)

        print(" e-con's Sample Python script for DepthVista ".center(100, " "))
        print(" Demonstrates the working of e-con's DepthVistaSDK ".center(100, " "))
        self.getSDKVersionResult(ctypes.byref(sdkMajorVer), ctypes.byref(sdkMinorVer1), ctypes.byref(sdkMinorVer2))
        sdk_version = f"{sdkMajorVer.value}.{sdkMinorVer1.value}.{sdkMinorVer2.value}"
        print(f" DepthVista SDK-Version = {sdk_version}".center(100, " "))

        self.main_menu_init()

    '''
        METHOD NAME : list_devices
        DESCRIPTION : Enumerates all the video devices connected and allows to select the device.
        RETURN      : device name, vendor id, product id, device path - if all the child functions are executes successfully
                     or None - if any of the child function is failed.
    '''
    def listDevices(self):
        #close and deinitialize device if already opened
        if(self.deviceSelected == True):
            self.deviceSelected = False
            if(self.closeDeviceResult(self.deviceHandle)):
                self.deviceHandle = DeviceHandle()
                cv2.destroyAllWindows()
        
        numberOfDevices = ctypes.c_uint32 ()
        if self.deviceCountResult(ctypes.byref(numberOfDevices)) == 1:
            numberOfDevices = numberOfDevices.value
            print(f'\nNumber of Depth_vista (TOF) Devices connected: {numberOfDevices}')
            print("\n\t0.Exit")

        #Getting device list information
        DeviceList = (DeviceInfo * numberOfDevices)()
        deviceNameValue = []
        vidValue = []
        pidValue = []
        devicePathValue = []
        serialNoValue = []

        for i in range(numberOfDevices):
            DeviceList[i].deviceName = b'None'
            DeviceList[i].pid = b'None'
            DeviceList[i].vid = b'None'
            DeviceList[i].devicePath = b'None'
            DeviceList[i].serialNo = b'None'
            DeviceList[i].devType = 0

            #Passing DeviceList By Reference by using byref
            if(self.getDeviceInfoResult(i ,ctypes.byref(DeviceList[i])) == 1):
                #Decoding the byte literals using decode()
                deviceNameValue.append(DeviceList[i].deviceName.decode('utf-8'))
                vidValue.append(DeviceList[i].vid.decode('utf-8'))
                pidValue.append(DeviceList[i].pid.decode('utf-8'))
                devicePathValue.append(DeviceList[i].devicePath.decode('utf-8'))
                serialNoValue.append(DeviceList[i].serialNo.decode('utf-8'))
                deviceType = DeviceList[i].devType

                print(f"\t{i + 1}.{deviceNameValue[i]}\t{serialNoValue[i]}")

        choice = get_integer("\nPick a Device to explore :\t", 0, numberOfDevices)

        if choice == 0:
            return None
        else:
            #Opening the device
            if(self.openDeviceResult(DeviceList[choice-1], ctypes.byref(self.deviceHandle)) < 1):
                print("Error in Opening the Device")
            else:
                self.deviceSelected = True
        return numberOfDevices, deviceNameValue[choice-1], vidValue[choice-1], pidValue[choice-1], devicePathValue[choice-1], serialNoValue[choice-1]

    '''
        METHOD NAME : update_colormap
        DESCRIPTION : Calls the updatecolormap method from depthvista SDK
    '''
    def update_colormap(self):
        global depth_min, depth_max, depth_range
        if(self.depthrange == 0):
            depth_min = 200
            depth_max = 1200
        elif(self.depthrange == 1):
            depth_min = 1000
            depth_max = 6000
        if(self.updateColorMapResult(self.deviceHandle, depth_min, depth_max + int(depth_range), 4)!=1):
            print("\nUpdate colormap failed")

    '''
        METHOD NAME : Preview
        DESCRIPTION : Provides the preview screen and render frames
    '''
    def Preview(self):
        global rgbdMappingflag
        while True:
            
            if(self.thread_lock_flag == True):
                self.thread_lock.acquire()

            #Thread end condition
            if(self.Thread_end.is_set()==True):
                cv2.destroyAllWindows()
                break
            
            #mode change loop
            while not self.modechange.is_set():

                self.save_possible = True
                
                if self.nextFrameResult(self.deviceHandle) == 1:
                    image_data_RGB = None
                    image_data_Depth = None
                    image_data_IR = None
                    #using GetFrames API to get all frames
                    if(self.getFramesResult(self.deviceHandle, ctypes.byref(self.allFrames)) == 1):
                        #rgb frame
                        if(self.datamode >= DataMode.Depth_IR_RGB_VGA_Mode.value):
                            rgb_height = self.allFrames.rgb.height
                            rgb_width = self.allFrames.rgb.width
                            #Acquiring thread lock to avoid race condition
                            lock = threading.Lock()
                            lock.acquire()
                            if bool(self.allFrames.rgb.frame_data):
                                expected_size = rgb_height * rgb_width * 2
                                actual_size = self.allFrames.rgb.size
                                if actual_size == expected_size:
                                    image_RGB = np.ctypeslib.as_array(self.allFrames.rgb.frame_data, (rgb_height, rgb_width, 2))
                                    image_data_RGB = cv2.cvtColor(image_RGB, cv2.COLOR_YUV2BGR_UYVY)
                                #Releasing thread lock
                                lock.release()
                                if(lock.locked() == 0):
                                    if(self.rgb_cap == True):
                                        #capture frame condition
                                        time=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                                        file_name = "DepthVista_rgb_"+time+".png"
                                        cv2.imwrite(file_name, image_data_RGB)
                                        self.rgb_cap=False
                                    #Stream show
                                    cv2.namedWindow("DepthVista_RGB_frame",cv2.WINDOW_GUI_NORMAL)
                                    cv2.setWindowProperty("DepthVista_RGB_frame", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_FULLSCREEN)
                                    cv2.imshow("DepthVista_RGB_frame",image_data_RGB)
                                    cv2.waitKey(1)

                        #Depth colormap frame
                        if(self.datamode != DataMode.IR_Mode.value) and (self.datamode <= DataMode.Depth_IR_RGB_HD_Mode.value):
                            self.update_colormap()
                            self.depthStreamStarted = True
                            #Acquiring thread lock to avoid race condition
                            lock = threading.Lock()
                            lock.acquire()                            
                            if bool(self.allFrames.depth_colormap.frame_data):
                                if ((rgbdMappingflag == True) and (self.datamode == DataMode.Depth_IR_RGB_HD_Mode.value)):
                                    expected_size = PRE_RGB_HD_HEIGHT * PRE_RGB_HD_WIDTH * 3
                                else:
                                    expected_size = PRE_DEPTH_HEIGHT * PRE_DEPTH_WIDTH * 3
                                actual_size = self.allFrames.depth_colormap.size
                                if actual_size == expected_size:
                                    image_data_Depth = np.ctypeslib.as_array(self.allFrames.depth_colormap.frame_data, (self.allFrames.depth_colormap.height, self.allFrames.depth_colormap.width, 3))

                                #Releasing thread lock
                                lock.release()

                                if(lock.locked() == 0):
                                    #capture frame condition
                                    if(self.depth_cap==True):
                                        time=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                                        file_name = "DepthVista_Depth_"+time+".bmp"
                                        cv2.imwrite(file_name, image_data_Depth)
                                        frame_data_uint16 = ctypes.cast(self.allFrames.raw_depth.frame_data, ctypes.POINTER(ctypes.c_uint16))
                                        capture_data = np.ctypeslib.as_array(frame_data_uint16, (self.allFrames.raw_depth.height, self.allFrames.raw_depth.width))
                                        file_name = "DepthVista_Raw_"+time+".raw"
                                        ply_file_name = "DepthVista_PLY_"+time+".ply"

                                        try:
                                            fp = open(file_name, 'wb+')
                                            fp.write(capture_data)
                                            fp.close()
                                            self.save_HD_ply_files(self.deviceCalibParams, capture_data, image_data_RGB, ply_file_name)                                          
                                        except IOError:
                                            print("File operation error.Image is not saved!")
                                            return False
                                        self.depth_cap=False
                                    #stream show
                                    cv2.namedWindow("DepthVista_Depthcolormap_frame",cv2.WINDOW_GUI_NORMAL)
                                    cv2.setWindowProperty("DepthVista_Depthcolormap_frame", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_FULLSCREEN)
                                    cv2.setMouseCallback("DepthVista_Depthcolormap_frame", self.mouseCallBck)
                                    cv2.imshow("DepthVista_Depthcolormap_frame",image_data_Depth)
                                    cv2.waitKey(1)

                        #IR frame
                        if(self.datamode != DataMode.Depth_Mode.value) and (self.datamode <=DataMode.Depth_IR_RGB_HD_Mode.value):
                            #Acquiring thread lock to avoid race condition
                            lock = threading.Lock()
                            lock.acquire()
                            if bool(self.allFrames.ir.frame_data):
                                expected_size = PRE_IR_HEIGHT * PRE_IR_WIDTH * 2
                                actual_size = self.allFrames.ir.size
                                if actual_size == expected_size:
                                    image_IR = ctypes.cast(self.allFrames.ir.frame_data, POINTER(ctypes.c_uint16))
                                    image_data_IR = np.ctypeslib.as_array(image_IR, (self.allFrames.ir.height, self.allFrames.ir.width, 1))
                                    #Releasing thread lock
                                    lock.release()
                                    if(lock.locked() == 0):
                                        #Stream show
                                        cv2.namedWindow("DepthVista_IR_frame",cv2.WINDOW_GUI_NORMAL)
                                        cv2.setWindowProperty("DepthVista_IR_frame", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_FULLSCREEN)
                                        cv2.imshow("DepthVista_IR_frame",image_data_IR)
                                        cv2.waitKey(1)
            
            self.depthStreamStarted=False
            self.save_possible = False
            cv2.destroyAllWindows()

    '''
        METHOD NAME : main_menu_init
        DESCRIPTION : Initiates the main menu to list available device(s)
    '''
    def main_menu_init(self):
        deviceInfo = self.listDevices()
        if deviceInfo is None:
            self.main_menu_exit()
        self.No_of_devices, self.device_name, self.vid, self.pid, self.device_path, self.serialNo = deviceInfo
        self.read_calibration_data()

        if(self.setdataModeResult(self.deviceHandle, DataMode.Depth_IR_RGB_HD_Mode.value) < 1):
            print("Datamode set failed")
        else:
            self.datamode=DataMode.Depth_IR_RGB_HD_Mode.value

        global depth_max, depth_min
        if(self.setDepthRangeResult(self.deviceHandle, DepthRange.FarRange.value) < 1):
            print("\nDepthRange set failed")
        else:
            depth_min = 1000
            depth_max = 6000
            self.depthrange = DepthRange.FarRange.value

        if(self.GetNextFrameThread.is_alive()):
            if(self.thread_lock_flag == True):
                self.thread_lock.release()
                self.thread_lock_flag = False
                self.optionsMenu()
            else:
                self.optionsMenu()

        global calibParamObtained, rgbdMappingflag
        if calibParamObtained:
            if(self.setRGBDMappingResult(self.deviceHandle,1) < 1):
                print("\nDepth RGB-D Mapping ON failed")
            else:
                rgbdMappingflag = True
        else:
            print("\nCalibration Data Not Found")
        self.GetNextFrameThread = threading.Thread(target=self.Preview, name="preview_thread", daemon=False)
        self.GetNextFrameThread.start()
        self.setavgregion(AvgRegion.CustomPtr.value)
        self.optionsMenu()

    '''
        METHOD NAME : optionsMenu
        DESCRIPTION : Lists the Options and gets the user input to transfer control
    '''
    def optionsMenu(self):
        optionsMenuOptions = {
            0: self.main_menu_exit,
            1: self.captureFrames,
        }
        while True:
            print(" OPTIONS ".center(100, "*"))
            print("\n\t0.Exit")
            print("\t1.Capture Frames\n")

            choice = get_integer("Pick a Relevant Option :\t", min(optionsMenuOptions, key=int),max(optionsMenuOptions, key=int))
            time.sleep(0.5)
            func = optionsMenuOptions.get(choice, lambda: "Invalid Selection")
            if not func():
                self.main_menu_exit()
   
    '''
        METHOD NAME : setavgregion
        DESCRIPTION : This method sets the avgregion
    '''
    def setavgregion(self, avgregion=ctypes.c_int16):
        if(self.setAvgRegionResult(self.deviceHandle, avgregion) < 1):
            print("Average Region set failed")

    '''
        METHOD NAME : captureFrames
        DESCRIPTION : Captures frame that is shown at the instance of call
    '''
    def captureFrames(self):
        global rgbdMappingflag
        if(self.save_possible == True):
            if(self.datamode != DataMode.IR_Mode.value) and (self.datamode <= DataMode.Depth_IR_RGB_HD_Mode.value):
                print("3D files saving might take some time. Please wait!!!")
                self.depth_cap = True
            if(self.datamode >= DataMode.Depth_IR_RGB_VGA_Mode.value):
                self.rgb_cap = True
            time.sleep(2)
            while 1:
                if(self.depth_cap == False) and (self.rgb_cap == False):
                    break
            print("\nFrame Capture Success")
        else:
            print("\nImage capture failed, Please start Preview Before Capturing Frames")
        self.optionsMenu()

    '''
        METHOD NAME : mouseCallBck
        DESCRIPTION : Sets position of the point from where depth value is to be taken
    '''  
    def mouseCallBck(self, event, x, y, flags, param):
        	if event == cv2.EVENT_LBUTTONDOWN:
                    livemouseptr = DepthPtr()
                    livemouseptr.X = x
                    livemouseptr.Y = y
                    if(self.setDepthPosResult(self.deviceHandle, livemouseptr)<1):
                        print("\n Failed to set mouse position")
    '''
        METHOD NAME : main_menu_exit
        DESCRIPTION : Exit call to close and deinitialize device
    '''
    def main_menu_exit(self):
        print("\nExit")
        if(self.GetNextFrameThread.is_alive()):
            self.modechange.set()
            self.Thread_end.set()
            if(self.thread_lock_flag == True):
                self.thread_lock.release()
            time.sleep(0.5)
            if(self.closeDeviceResult(self.deviceHandle)==1):
                if(self.deinitializeResult()==1):
                    cv2.destroyAllWindows()
        exit(0)

    def read_calibration_data(self):
        global calibParamObtained

        if not calibParamObtained:
            self.getDevCalibParamsResult(self.deviceHandle, ctypes.byref(self.deviceCalibParams))
            calibParamObtained = True

    def save_HD_ply_files(self, calibParams, Depth, rgbFrame, fileName):
        global depth_min, depth_max, depth_range

        if calibParams is None or Depth is None:
            return -2
        
        # Getting the fx, fy, cx and cy from HD RGB intrinsic.
        focalLengthx = self.deviceCalibParams.rgbCamHDIntrinsic.fx
        focalLengthy = self.deviceCalibParams.rgbCamHDIntrinsic.fy
        principlePointx = self.deviceCalibParams.rgbCamHDIntrinsic.cx
        principlePointy = self.deviceCalibParams.rgbCamHDIntrinsic.cy

        DepthFloat = np.float32(Depth)
        zoom_factor = 1.0

        points = []
        for v in range(DepthFloat.shape[0]):
            for u in range(DepthFloat.shape[1]):
                Z = DepthFloat[v, u] / zoom_factor
                raw_depth = Depth[v, u]
                if raw_depth > depth_max or raw_depth < depth_min:
                    continue

                z = Z
                x = (u - principlePointx) * Z / focalLengthx
                y = ((v - principlePointy) * Z / focalLengthy)

                pclPoint = {
                    'xyz': [x / 1000, -y / 1000, -z / 1000],  # Assuming METER_TO_MILLIMETER = 1000
                    'rgb': rgbFrame[v, u][::-1]  # BGR to RGB
                }

                points.append(pclPoint)

        if not points:
            return -1

        # Save to the ply file
        PLY_START_HEADER = "ply"
        PLY_END_HEADER = "end_header"
        PLY_ASCII = "format ascii 1.0"
        PLY_ELEMENT_VERTEX = "element vertex"

        with open(fileName, 'w') as ofs:
            ofs.write(PLY_START_HEADER + '\n')
            ofs.write(PLY_ASCII + '\n')
            ofs.write(PLY_ELEMENT_VERTEX + ' ' + str(len(points)) + '\n')
            ofs.write("property float x\n")
            ofs.write("property float y\n")
            ofs.write("property float z\n")
            ofs.write("property uchar red\n")
            ofs.write("property uchar green\n")
            ofs.write("property uchar blue\n")
            ofs.write(PLY_END_HEADER + '\n')

        with open(fileName, 'a') as ofs_text:
            for pclPoint in points:
                ofs_text.write(f"{pclPoint['xyz'][0]} {pclPoint['xyz'][1]} {pclPoint['xyz'][2]} ")
                ofs_text.write(f"{pclPoint['rgb'][0]} {pclPoint['rgb'][1]} {pclPoint['rgb'][2]}\n")

if __name__ == '__main__':
    main = MainClass()
