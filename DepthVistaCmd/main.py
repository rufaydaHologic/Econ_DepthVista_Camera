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
depth_intrinsic = None
rgb_intrinsic = None
rgbHD_intrinsic = None
rgbdMappingflag = False
hdCalibration = False
captureDone = False
PRE_RGB_VGA_WIDTH = PRE_DEPTH_WIDTH = PRE_IR_WIDTH = 640
PRE_RGB_VGA_HEIGHT = PRE_DEPTH_HEIGHT = PRE_IR_HEIGHT = 480

#Preview RGB HD Resolution 1280x720
PRE_RGB_HD_WIDTH = 1280
PRE_RGB_HD_HEIGHT = 720

# Preview RGB Full HD Resolution 1920x1080
PRE_RGB_FULL_HD_WIDTH = 1920
PRE_RGB_FULL_HD_HEIGHT = 1080

#Preview RGB original Resolution 1920x1200
PRE_RGB_ORIGINAL_WIDTH = 1920
PRE_RGB_ORIGINAL_HEIGHT = 1200

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
    Defining Datamode Enum
'''
class DataMode(Enum):
    Depth_IR_Mode = 0
    Depth_IR_Conf_Mode = 1
    Depth_Mode = 2
    IR_Mode = 3
    Depth_IR_RGB_VGA_Mode = 4
    Depth_IR_RGB_HD_Mode = 5
    RGB_VGA_Mode = 6
    RGB_HD_Mode = 7
    RGB_Full_HD_Mode = 8
    RGB_1200p_Mode = 9
    Depth_RGB_HD_Mode = 11
    RGB_960x540_Mode_100fps = 14
    IR_RGB_HD_Mode = 18
    RGB_4K_Mode_15fps = 19

'''
    Defining DepthRange Enum
'''
class DepthRange(Enum):
    NearRange = 0
    FarRange = 1

'''
    Defining FrameType Enum
'''
class FrameType(Enum):
    IRPreviewFrame = 0
    DepthColorMap = 1
    RGBFrame = 2
    DepthRawFrame = 3
    ConfidenceFrame = 4

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

    setdataMode = depthVistaLib.SetDataMode
    setdataMode.restype = ctypes.c_int
    setdataMode.argtypes = [DeviceHandle, ctypes.c_int32]

    getDataMode = depthVistaLib.GetDataMode
    getDataMode.restype = ctypes.c_int
    getDataMode.argtypes = [DeviceHandle, ctypes.POINTER(ctypes.c_int32) ]

    setDepthRange = depthVistaLib.SetDepthRange
    setDepthRange.restype = ctypes.c_int
    setDepthRange.argtypes = [DeviceHandle, ctypes.c_uint16]

    getDepthRange = depthVistaLib.GetDepthRange
    getDepthRange.restype = ctypes.c_int
    getDepthRange.argtypes = [DeviceHandle, ctypes.POINTER(ctypes.c_int16) ]

    nextFrameResult = depthVistaLib.GetNextFrame
    nextFrameResult.restype = ctypes.c_int
    nextFrameResult.argtypes = [ DeviceHandle ]

    ToFFrameResult = depthVistaLib.GetToFFrame
    ToFFrameResult.restype = ctypes.c_int
    ToFFrameResult.argtypes = [DeviceHandle, ctypes.c_uint32, ctypes.POINTER(tofFrame)]

    getFramesResult = depthVistaLib.GetFrames
    getFramesResult.restype = ctypes.c_int
    getFramesResult.argtypes = [DeviceHandle, ctypes.POINTER(frames)]
    
    updateColorMapRes = depthVistaLib.UpdateColorMap
    updateColorMapRes.restype = ctypes.c_int
    updateColorMapRes.argtypes = [DeviceHandle, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

    setRGBDMapping = depthVistaLib.SetRGBDMapping
    setRGBDMapping.restype = ctypes.c_int
    setRGBDMapping.argtypes = [DeviceHandle, ctypes.c_uint16]

    setAvgRegion = depthVistaLib.SetAvgRegion
    setAvgRegion.restype = ctypes.c_int
    setAvgRegion.argtypes = [DeviceHandle, ctypes.c_uint16]

    getSDKVersion = depthVistaLib.GetSDKVersion
    getSDKVersion.restype = ctypes.c_int
    getSDKVersion.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16)]

    readfirmwareVersion = depthVistaLib.ReadFirmwareVersion
    readfirmwareVersion.restype = ctypes.c_int
    readfirmwareVersion.argtypes = [DeviceHandle, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16)]

    getDepthIRValues = depthVistaLib.GetDepthIRValues
    getDepthIRValues.restype = ctypes.c_int
    getDepthIRValues.argtypes = [DeviceHandle, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

    setDepthPos = depthVistaLib.SetDepthPos
    setDepthPos.restype = ctypes.c_int
    setDepthPos.argtypes = [DeviceHandle, DepthPtr]

    getUniqueID = depthVistaLib.GetUniqueID
    getUniqueID.restype = ctypes.c_int
    getUniqueID.argtypes = [DeviceHandle, ctypes.POINTER(ctypes.c_uint64)]

    calibReadReqDepthIntrinsicRes = depthVistaLib.CalibReadReqDepthIntrinsic
    calibReadReqDepthIntrinsicRes.restype = ctypes.c_int
    calibReadReqDepthIntrinsicRes.argtypes = [DeviceHandle, ctypes.POINTER(c_int)]

    calibReadDepthInstrinsicRes = depthVistaLib.CalibReadDepthIntrinsic
    calibReadDepthInstrinsicRes.restype = ctypes.c_int
    calibReadDepthInstrinsicRes.argtypes = [DeviceHandle, ctypes.POINTER(c_int), ctypes.POINTER(c_ubyte)]

    getSDKVersionResult = depthVistaLib.GetSDKVersion
    getSDKVersionResult.restype = ctypes.c_int
    getSDKVersionResult.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16)]

    Thread_end = threading.Event()
    modechange = threading.Event()

    ToFFrameDepth = tofFrame()
    ToFFrameRGB = tofFrame()
    ToFFrameIR = tofFrame()

    allFrames = frames()

    GetNextFrameThread = threading.Thread()
    
    thread_lock_flag = False
    thread_lock = threading.Lock()

    deviceHandle = DeviceHandle()
    datamode = DataMode.Depth_IR_RGB_VGA_Mode.value
    depthrange = DepthRange.FarRange.value

    depth_cap = False
    rgb_cap = False
    IR_cap = False

    deviceSelected = False
    
    depthStreamStarted = False

    save_possible = False

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
        self.deviceType = None
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
            if(self.closeDeviceResult(self.deviceHandle) == 1):
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
        if(self.updateColorMapRes(self.deviceHandle, depth_min, depth_max + int(depth_range), 4)!=1):
            print("\nUpdate colormap failed")

    '''
        METHOD NAME : Preview
        DESCRIPTION : Provides the preview screen and render frames
    '''
    def Preview(self):
        global depth_intrinsic, rgb_intrinsic, rgbHD_intrinsic, rgbdMappingflag
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
                            frametype = FrameType.RGBFrame.value

                            if(self.datamode == DataMode.RGB_VGA_Mode.value) or (self.datamode == DataMode.Depth_IR_RGB_VGA_Mode.value):
                                rgb_height = PRE_RGB_VGA_HEIGHT
                                rgb_width = PRE_RGB_VGA_WIDTH
                            if(self.datamode == DataMode.RGB_1200p_Mode.value):
                                rgb_height = PRE_RGB_ORIGINAL_HEIGHT
                                rgb_width = PRE_RGB_ORIGINAL_WIDTH
                            if(self.datamode == DataMode.RGB_HD_Mode.value) or (self.datamode == DataMode.Depth_IR_RGB_HD_Mode.value):
                                rgb_height = PRE_RGB_HD_HEIGHT
                                rgb_width = PRE_RGB_HD_WIDTH
                            if(self.datamode == DataMode.RGB_Full_HD_Mode.value):
                                rgb_height = PRE_RGB_FULL_HD_HEIGHT
                                rgb_width = PRE_RGB_FULL_HD_WIDTH

                            #Acquiring thread lock to avoid race condition
                            lock = threading.Lock()
                            lock.acquire()

                            # commented below GetTofFrame API call to demonstrate GetFrames API
                            # if(self.ToFFrameResult(self.deviceHandle, frametype, ctypes.byref(self.ToFFrameRGB)) == 1):
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
                            frametype = FrameType.DepthColorMap.value
                            self.update_colormap()
                            self.depthStreamStarted = True

                            #Acquiring thread lock to avoid race condition
                            lock = threading.Lock()
                            lock.acquire()
                            
                            # commented below GetTofFrame API call to demonstrate GetFrames API
                            # if(self.ToFFrameResult(self.deviceHandle, frametype, ctypes.byref(self.ToFFrameDepth)) == 1):
                            if bool(self.allFrames.depth_colormap.frame_data):
                                if ((rgbdMappingflag == True) and (self.datamode == DataMode.Depth_IR_RGB_HD_Mode.value)):
                                    expected_size = PRE_RGB_HD_HEIGHT * PRE_RGB_HD_WIDTH * 3
                                else:
                                    expected_size = PRE_DEPTH_HEIGHT * PRE_DEPTH_WIDTH * 3
                                actual_size = self.allFrames.depth_colormap.size
                                if actual_size == expected_size:
                                    if ((rgbdMappingflag == True) and (self.datamode == DataMode.Depth_IR_RGB_HD_Mode.value)):
                                        image_data_Depth = np.ctypeslib.as_array(self.allFrames.depth_colormap.frame_data, (PRE_RGB_HD_HEIGHT, PRE_RGB_HD_WIDTH, 3))
                                    else:
                                        image_data_Depth = np.ctypeslib.as_array(self.allFrames.depth_colormap.frame_data, (PRE_DEPTH_HEIGHT, PRE_DEPTH_WIDTH, 3))

                                #Releasing thread lock
                                lock.release()

                                if(lock.locked() == 0):
                                    #capture frame condition
                                    if(self.depth_cap==True):
                                        time=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                                        file_name = "DepthVista_Depth_"+time+".bmp"
                                        cv2.imwrite(file_name, image_data_Depth)
                                        # commented below GetTofFrame API call to demonstrate GetFrames API
                                        # self.ToFFrameResult(self.deviceHandle, FrameType.DepthRawFrame.value, ctypes.byref(self.ToFFrameDepth))
                                        frame_data_uint16 = ctypes.cast(self.allFrames.raw_depth.frame_data, ctypes.POINTER(ctypes.c_uint16))
                                        capture_data = np.ctypeslib.as_array(frame_data_uint16, (PRE_DEPTH_HEIGHT, PRE_DEPTH_WIDTH))
                                        file_name = "DepthVista_Raw_"+time+".raw"
                                        ply_file_name = "DepthVista_PLY_"+time+".ply"

                                        try:
                                            fp = open(file_name, 'wb+')
                                            fp.write(capture_data)
                                            fp.close()
                                            if rgbdMappingflag:
                                                if self.datamode == DataMode.Depth_IR_RGB_HD_Mode.value and hdCalibration == True:
                                                    self.save_ply_files(rgbHD_intrinsic, capture_data, image_data_RGB, ply_file_name)
                                                else:
                                                    self.save_ply_files(rgb_intrinsic, capture_data, image_data_RGB, ply_file_name)
                                            else:
                                                self.save_ply_files(depth_intrinsic, capture_data, image_data_Depth, ply_file_name)                                            
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
                            frametype = FrameType.IRPreviewFrame.value

                            #Acquiring thread lock to avoid race condition
                            lock = threading.Lock()
                            lock.acquire()

                            # commented below GetTofFrame API call to demonstrate GetFrames API
                            # if(self.ToFFrameResult(self.deviceHandle, frametype, ctypes.byref(self.ToFFrameIR)) == 1):
                            if bool(self.allFrames.ir.frame_data):
                                expected_size = PRE_IR_HEIGHT * PRE_IR_WIDTH * 2
                                actual_size = self.allFrames.ir.size
                                if actual_size == expected_size:
                                    image_IR = ctypes.cast(self.allFrames.ir.frame_data, POINTER(ctypes.c_uint16))
                                    image_data_IR = np.ctypeslib.as_array(image_IR, (PRE_IR_HEIGHT, PRE_IR_WIDTH, 1))

                                    #Releasing thread lock
                                    lock.release()

                                    if(lock.locked() == 0):
                                        #capture frame condition
                                        if(self.IR_cap == True):
                                            time=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                                            file_name = "DepthVista_IR_"+time+".png"
                                            cv2.imwrite(file_name, image_data_IR)
                                            self.IR_cap=False
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

        if(self.GetNextFrameThread.is_alive()):
            if(self.thread_lock_flag == True):

                self.datamode=self.getdatamode()
                time.sleep(0.5)
                self.depthrange=self.getdepthrange()
                time.sleep(0.5)
                self.modechange.clear()
                self.thread_lock.release()
                self.thread_lock_flag = False
                self.propertiesMenu()
            else:
                self.modechange.set()
                self.datamode=self.getdatamode()
                time.sleep(0.5)
                self.depthrange=self.getdepthrange()
                time.sleep(0.5)
                self.modechange.clear()
                self.propertiesMenu()

        self.datamode=self.getdatamode()
        time.sleep(0.5)
        self.depthrange=self.getdepthrange()
        time.sleep(0.5)
        self.modechange.clear()
        self.GetNextFrameThread = threading.Thread(target=self.Preview, name="preview_thread", daemon=False)
        self.GetNextFrameThread.start()

        self.setavgregion(AvgRegion.CustomPtr.value)

        self.propertiesMenu()

    '''
        METHOD NAME : propertiesMenu
        DESCRIPTION : Lists the properties and gets the user input to transfer control
    '''
    def propertiesMenu(self):
        propertiesMenuOptions = {
            0: self.main_menu_exit,
            1: self.listDevices,
            2: self.streamingMode,
            3: self.depthRange,
            4: self.rgbDMapping,
            5: self.captureFrames,
            6: self.uniqueId,
            7: self.readFirmwareVersion,
            8: self.getDepthValue,
        }
        while True:
            print("CAMERA PROPERTIES MENU".center(100, "*"))
            print("\n\t0.Exit")
            print("\t1.Back")
            print("\t2.Streaming Mode")
            print("\t3.Depth Range")
            print("\t4.RGB-D Mapping")
            print("\t5.Capture Frames")
            print("\t6.Unique Id")
            print("\t7.Read Firmware Version")
            print("\t8.Get Depth value\n")

            choice = get_integer("Pick a Relevant Type Of Camera Properties :\t", min(propertiesMenuOptions, key=int),max(propertiesMenuOptions, key=int))
            time.sleep(0.5)
            func = propertiesMenuOptions.get(choice, lambda: "Invalid Selection")

            if not func():
                self.main_menu_exit()

    '''
        METHOD NAME : streamingMode
        DESCRIPTION : Lists the streaming modes available and gets the user input to transfer control
    '''    
    def streamingMode(self):
        
        streamingModeOptions={
            0: self.main_menu_exit,
            1: self.propertiesMenu,
            2: self.propertiesMenu,
            3: DataMode.Depth_IR_Mode,
            4: DataMode.Depth_Mode,
            5: DataMode.IR_Mode,
            6: DataMode.Depth_IR_RGB_VGA_Mode,
            7: DataMode.Depth_IR_RGB_HD_Mode,
            8: DataMode.RGB_VGA_Mode,
            9: DataMode.RGB_HD_Mode,
            10: DataMode.RGB_Full_HD_Mode,
            11: DataMode.RGB_1200p_Mode,
        }

        while True:
            print("\nTotal Number of Streaming Modes Supported By the Camera: 9")
            print("STREAMING MODE MENU".center(100, "*"))
            print("\n\t0.Exit")
            print("\t1.Back")
            print("\t2.Main Menu")
            print("\t3.Depth IR Mode")
            print("\t4.Depth Mode")
            print("\t5.IR Mode")
            print("\t6.Depth IR RGB(VGA) Mode")
            print("\t7.Depth IR RGB(HD) Mode")
            print("\t8.RGB(VGA) Mode")
            print("\t9.RGB(HD) Mode")
            print("\t10.RGB(Full HD) Mode")
            print("\t11.RGB(1200p) Mode")

            choice = get_integer("\nPick a relevant Streaming Mode :\t", min(streamingModeOptions, key=int),max(streamingModeOptions, key=int))
            func = streamingModeOptions.get(choice, lambda: "\nInvalid Selection")
            
            if(choice > 2)and(choice < 12):

                if(self.GetNextFrameThread.is_alive()):
                    if(self.thread_lock_flag == True):

                        self.setdatamode(func.value)
                        self.datamode=self.getdatamode()
                        time.sleep(0.5)
                        self.modechange.clear()
                        self.thread_lock.release()
                        self.thread_lock_flag = False
                        self.propertiesMenu()
                    else:
                        self.modechange.set()
                        self.setdatamode(func.value)
                        self.datamode=self.getdatamode()
                        time.sleep(0.5)
                        self.modechange.clear()
                        self.propertiesMenu()

                self.setdatamode(func.value)
                self.datamode=self.getdatamode()
                time.sleep(0.5)
                self.modechange.clear()
                self.GetNextFrameThread = threading.Thread(target=self.Preview, name="preview_thread", daemon=False)
                self.GetNextFrameThread.start()
                self.propertiesMenu()

            if(choice < 3)and(choice >= 0):
                if not func():
                    self.propertiesMenu()

    '''
        METHOD NAME : getdatamode
        DESCRIPTION : This method returns the value of active datamode
        RETURN      : Integer value of Datamode
    '''
    def getdatamode(self):
        datamode=ctypes.c_int32()
        if(self.getDataMode(self.deviceHandle, ctypes.byref(datamode)) == 1):
            return datamode.value
        else:
            print("Datamode get failed")

    '''
        METHOD NAME : setdatamode
        DESCRIPTION : This method sets the datamode
    '''
    def setdatamode(self, datamode=ctypes.c_int32):
        if(self.setdataMode(self.deviceHandle, datamode) < 1):
            print("Datamode set failed")

    def getdepthrange(self):
        depthrange=ctypes.c_int16()
        if(self.getDepthRange(self.deviceHandle, ctypes.byref(depthrange)) == 1):
            return depthrange.value
        else:
            print("DepthRange get failed")
   
    '''
        METHOD NAME : setavgregion
        DESCRIPTION : This method sets the avgregion
    '''
    def setavgregion(self, avgregion=ctypes.c_int16):
        if(self.setAvgRegion(self.deviceHandle, avgregion) < 1):
            print("Average Region set failed")

    '''
        METHOD NAME : depthRange
        DESCRIPTION : Lists the depth range available and gets the user input to transfer control
    '''
    def depthRange(self):
        depthRangeOptions={
            0: self.main_menu_exit,
            1: self.propertiesMenu,
            2: self.propertiesMenu,
            3: self.nearMode,
            4: self.farMode,
        }
        while True:
            print("DEPTH_RANGE MENU".center(100, "*"))
            print("\n\t0.Exit")
            print("\t1.Back")
            print("\t2.Main Menu")
            print("\t3.Near Mode")
            print("\t4.Far Mode")

            depthRangechoice = get_integer("\nPick a Relevant Depth Range :\t", min(depthRangeOptions, key=int),max(depthRangeOptions, key=int))
            func = depthRangeOptions.get(depthRangechoice, lambda: "Invalid Selection")

            if not func():
                self.main_menu_exit()

    '''
        METHOD NAME : nearMode
        DESCRIPTION : Change the depth range to near range
    '''
    def nearMode(self):
        global depth_max, depth_min
        if(self.setDepthRange(self.deviceHandle, 0) < 1):
            print("\nNear mode set failed")
        else:
            depth_min = 200
            depth_max = 1200
            self.depthrange = 0
            print("\nDepth Range Near")
        self.propertiesMenu()

    '''
        METHOD NAME : farMode
        DESCRIPTION : Change the depth range to far range
    '''
    def farMode(self):
        global depth_max, depth_min
        if(self.setDepthRange(self.deviceHandle, 1) < 1):
            print("\nFar mode set failed")
        else:
            depth_min = 1000
            depth_max = 6000
            self.depthrange = 1
            print("\nDepth Range Far")
        self.propertiesMenu()

    '''
        METHOD NAME : rgbDMapping
        DESCRIPTION : Lists the distortion mode options available and gets the user input to transfer control
    '''
    def rgbDMapping(self):
        rgbDMappingMenuOptions = {
            0: self.main_menu_exit,
            1: self.propertiesMenu,
            2: self.__init__,
            3: self.rgbDMappingOff,
            4: self.rgbDMappingOn,
            }
        while True:
            print("DEPTH RGB-D MAPPING MODE MENU".center(100, "*"))
            print("\n\t0.Exit")
            print("\t1.Back")
            print("\t2.Main Menu")
            print("\t3.RGB-D Mapping OFF")
            print("\t4.RGB-D Mapping ON")

            choice = get_integer("\nPick a relevant RGB-D Mapping mode :\t", min(rgbDMappingMenuOptions, key=int),max(rgbDMappingMenuOptions, key=int))
            func = rgbDMappingMenuOptions.get(choice, lambda: "Invalid Selection")

            if not func():
                self.main_menu_exit()

    '''
        METHOD NAME : rgbDMappingOff
        DESCRIPTION : Turns OFF RGB-D Mapping at the instance of call
    '''
    def rgbDMappingOff(self):
        global rgbdMappingflag
        if(self.setRGBDMapping(self.deviceHandle,0) < 1):
            print("\nDepth RGB-D Mapping OFF failed")
        else:
            rgbdMappingflag = False
            print("\nDepth RGB-D Mapping OFF")
        self.propertiesMenu()

    '''
        METHOD NAME : rgbDMappingOn
        DESCRIPTION : Turns ON RGB-D Mapping at the instance of call
    '''
    def rgbDMappingOn(self):
        global calibParamObtained, rgbdMappingflag
        if calibParamObtained:
            if(self.setRGBDMapping(self.deviceHandle,1) < 1):
                print("\nDepth RGB-D Mapping ON failed")
            else:
                rgbdMappingflag = True
                print("\nDepth RGB-D Mapping ON")
        else:
            print("\nCalibration Data Not Found")
        self.propertiesMenu()

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
            if(self.datamode != DataMode.Depth_Mode.value) and (self.datamode <=DataMode.Depth_IR_RGB_HD_Mode.value):
                self.IR_cap = True
            time.sleep(2)
            while 1:
                if(self.depth_cap == False) and (self.rgb_cap == False) and (self.IR_cap == False):
                    break
            print("\nFrame Captured")
        else:
            print("\nImage capture failed, Please start Preview Before Capturing Frames")
        self.propertiesMenu()

    '''
        METHOD NAME : uniqueId
        DESCRIPTION : Prints the uniqueID of the camera
    '''
    def uniqueId(self):
        uniqueID = ctypes.c_uint64()
        
        if(self.getUniqueID(self.deviceHandle, ctypes.byref(uniqueID)) < 1):
            print("\nUnique ID read failed")
        else:
            print("\nUnique ID of the Camera is ",uniqueID.value)
        self.propertiesMenu()

    '''
        METHOD NAME : readFirmwareVersion
        DESCRIPTION : Prints the firmware version
    '''
    def readFirmwareVersion(self):
        gMajorVersion = ctypes.c_uint8()
        gMinorVersion1 = ctypes.c_uint8()
        gMinorVersion2 = ctypes.c_uint16()
        gMinorVersion3 = ctypes.c_uint16()
    
        if(self.readfirmwareVersion(self.deviceHandle, ctypes.byref(gMajorVersion), ctypes.byref(gMinorVersion1), ctypes.byref(gMinorVersion2), ctypes.byref(gMinorVersion3)) < 1):
            print("\nFirmware version Read failed")
        else:
            FWV = '.'.join(str(x) for x in [gMajorVersion.value, gMinorVersion1.value, gMinorVersion2.value, gMinorVersion3.value])
            print("\nFirmware version :\t", FWV)

        self.propertiesMenu()

    '''
        METHOD NAME : mouseCallBck
        DESCRIPTION : Sets position of the point from where depth value is to be taken
    '''  
    def mouseCallBck(self, event, x, y, flags, param):
        	if event == cv2.EVENT_LBUTTONDOWN:
                    livemouseptr = DepthPtr()
                    livemouseptr.X = x
                    livemouseptr.Y = y
                    if(self.setDepthPos(self.deviceHandle, livemouseptr)<1):
                        print("\n Failed to set mouse position")

    '''
        METHOD NAME : getDepthValue
        DESCRIPTION : Lists the Depth value options available and gets the user input to transfer control
    '''  
    def getDepthValue(self):
        depthvalueMenuOptions = {
            0: self.main_menu_exit,
            1: self.propertiesMenu,
            2: self.propertiesMenu,
            3: self.getdepthvaluecentre,
            4: self.getdepthvaluecustom,
            }
        
        if(self.depthStreamStarted):
            while True:
                print("GET DEPTH VALUE MENU".center(100, "*"))
                print("\n\t0.Exit")
                print("\t1.Back")
                print("\t2.Main Menu")
                print("\t3.Centre")
                print("\t4.Custom Co-ordinate")

                choice = get_integer("\nPick a relevant Position :\t", min(depthvalueMenuOptions, key=int),max(depthvalueMenuOptions, key=int))
                func = depthvalueMenuOptions.get(choice, lambda: "Invalid Selection")

                if not func():
                    self.main_menu_exit()

        else:
            print("\nStart Depth Stream to get Depth value")
            self.propertiesMenu()

    '''
        METHOD NAME : getdepthvaluecentre
        DESCRIPTION : Prints the average depth value of centre of the frame
    '''
    def getdepthvaluecentre(self):

        ptr = DepthPtr()

        ptr.X = ctypes.c_int(int(320)).value
        ptr.Y = ctypes.c_int(int(240)).value

        if(self.setDepthPos(self.deviceHandle, ptr) < 1):
            print("\n Failed to set position")
        else:
            avgDepth = ctypes.c_int()
            stdDepth = ctypes.c_int()
            avgIR = ctypes.c_int()
            stdIR = ctypes.c_int()
            if(self.getDepthIRValues(self.deviceHandle, ctypes.byref(avgDepth), ctypes.byref(stdDepth), ctypes.byref(avgIR), ctypes.byref(stdIR)) < 1):
                print("\nFailed to get Depth value")
            else:
                print("\nDepth value :\t", avgDepth.value)
        self.propertiesMenu()

    '''
        METHOD NAME : getdepthvaluecustom
        DESCRIPTION : Prints the average depth value of the given position
    '''
    def getdepthvaluecustom(self):

        ptr = DepthPtr()

        x_pos = get_integer("\nEnter X co-ordinate between the range 17 to 624 :\t", 16,624, "\nInvalid Input")
        y_pos = get_integer("\nEnter Y co-ordinate between the range 17 to 464 :\t", 16,464, "\nInvalid Input")

        ptr.X = ctypes.c_int(int(x_pos)).value
        ptr.Y = ctypes.c_int(int(y_pos)).value

        if(self.setDepthPos(self.deviceHandle, ptr) < 1):
            print("\n Failed to set position")
        else:
            avgDepth = ctypes.c_int()
            stdDepth = ctypes.c_int()
            avgIR = ctypes.c_int()
            stdIR = ctypes.c_int()
            if(self.getDepthIRValues(self.deviceHandle, ctypes.byref(avgDepth), ctypes.byref(stdDepth), ctypes.byref(avgIR), ctypes.byref(stdIR)) < 1):
                print("\nFailed to get Depth value")
            else:
                print("\nDepth value :\t", avgDepth.value)
        self.propertiesMenu()

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
        global depth_intrinsic, rgb_intrinsic, rgbHD_intrinsic, calibParamObtained
        read_length_depth = ctypes.c_int()

        if not calibParamObtained:
            readDepthIntrinsic = cv2.FileStorage()
            readDepthIntrinsic.open("depth_intrinsic.yml", cv2.FileStorage_READ)

            if not readDepthIntrinsic.isOpened():
                print("depth_intrinsic.yml is not found, please make sure the file is in the directory")
            else:
                sccm_d_node = readDepthIntrinsic.getNode("SCCM_D")
                if sccm_d_node.isNone():
                    print("SCCM_D parameter is not present in the file")
                else:
                    depth_intrinsic = sccm_d_node.mat()

            readRGBIntrinsic = cv2.FileStorage()
            readRGBIntrinsic.open("rgb_intrinsic.yml", cv2.FileStorage_READ)

            if not readRGBIntrinsic.isOpened():
                print("rgb_intrinsic.yml is not found, please make sure the file is in the directory")
            else:
                sccm_rgb_node = readRGBIntrinsic.getNode("SCCM_RGB")
                if sccm_rgb_node.isNone():
                    print("SCCM_RGB parameter is not present in the file")
                else:
                    rgb_intrinsic = sccm_rgb_node.mat()

            if rgb_intrinsic is not None and rgb_intrinsic.shape[0] > 0 and rgb_intrinsic.shape[1] > 0:
                sccm_rgbHD_node = readRGBIntrinsic.getNode("SCCM_RGB_HD")
                if sccm_rgbHD_node.isNone():
                    rgbHD_intrinsic = np.array([
                        [rgb_intrinsic[0, 0] * 1.67, rgb_intrinsic[0, 1], rgb_intrinsic[0, 2] * 2.00],
                        [rgb_intrinsic[1, 0], rgb_intrinsic[1, 1] * 1.67, rgb_intrinsic[1, 2] * 1.5],
                        [rgb_intrinsic[2, 0], rgb_intrinsic[2, 1], rgb_intrinsic[2, 2]]
                    ])
                else:
                    rgbHD_intrinsic = sccm_rgbHD_node.mat()
            else:
                print("Invalid or missing rgb_intrinsic")

        if rgb_intrinsic is not None and rgb_intrinsic.shape[0] > 0 and rgb_intrinsic.shape[1] > 0:
            rgbIntrinsicData = True
        else:
            rgbIntrinsicData = False

        if rgbHD_intrinsic is not None and rgbHD_intrinsic.shape[0] > 0 and rgbHD_intrinsic.shape[1] > 0:
            hdCalibration = True
        else:
            hdCalibration = False

        if depth_intrinsic is not None and depth_intrinsic.shape[0] > 0 and depth_intrinsic.shape[1] > 0:
            depthIntrinsicData = True
        else:
            depthIntrinsicData = False

        if rgbIntrinsicData and depthIntrinsicData:
            calibParamObtained = True

    def save_ply_files(self, Intrinsic, Depth, rgbFrame, fileName):
        global depth_min, depth_max, depth_range

        if Intrinsic is None or Depth is None:
            return -2

        focalLengthx = Intrinsic[0, 0]
        focalLengthy = Intrinsic[1, 1]
        principlePointx = Intrinsic[0, 2]
        principlePointy = Intrinsic[1, 2]

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
