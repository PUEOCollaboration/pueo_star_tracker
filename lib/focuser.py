# Standard Imports
import logging
import serial
import time

from lib.star_comm_bridge import DummyPueoServer
from lib.common import logit, get_dt

class Focuser:
    """
    TODO: Make sure only one command is running at the same TIME!!!
    """
    f_stop_sequence = [
        'f14', 'f15', 'f17', 'f18', 'f20', 'f22', 'f24', 'f26', 'f28', 'f30',
        'f33', 'f36', 'f40', 'f43', 'f48', 'f52', 'f56', 'f61', 'f67', 'f73',
        'f80', 'f87', 'f95', 'f103', 'f113', 'f123', 'f134', 'f146', 'f160'
    ]

    aperture_position = None    # opened, closed
    focus_inf = 10000
    _focus_position: int = None  # 0, 1, ...
    _aperture_pos: int = None  # 0, 1, .. 28
    _aperture_f_val: str = None  # f14, ...

    def __init__(self, cfg):
        self.cfg = cfg
        self.log = logging.getLogger('pueo')
        self.log.info('Initialising Focuser')
        self.aperture_position = 'closed'

        StubClass = type('StubClass', (), {})
        self.server = StubClass()
        self.server.write = print

        # def initialize_camera_focuser(self):
        # port1 = '/dev/ttyUSB2'  # focuser port
        # port1 = # focuser_port: '/dev/ttyUSB0'  # focuser port
        # port2 = # computer_port: '/dev/ttyUSB1'  # computer comm port
        # baud = # baud_rate = 115200

        self.ser = None
        self.f_stops = []
        max_retry = 6
        retry = 0
        status = False
        baud_rates = [self.cfg.baud_rate, 115200, 57600, 38400, 19200, 9600]
        while retry < max_retry:
            try:
                br_idx = min(retry, len(baud_rates) - 1) # Just in case - Safe indexing
                baud_rate = baud_rates[br_idx]
                if retry == 0:
                    self.log.info(f'Initialising focuser: baud_rate: {baud_rate}')
                else:
                    self.log.warning(f'Initialising focuser: retry: {retry} baud_rate: {baud_rate}')

                self.ser = serial.Serial(self.cfg.focuser_port, baudrate=baud_rate, timeout=2)  # camera focuser port
               # TODO: Refactor
                # Another Computer for sending commands via serial - this will be replaced by GUI!
                # other computer serial port
                # self.ser2 = serial.Serial(self.cfg.computer_port, self.cfg.baud_rate, timeout=1)

                self.f_min = None
                self.num_stops: int = 0
                self.f_max = None
                # self.f_stops = []
                self.response_mode(verbose_mode=1, protocol_mode=0)  # Setting Verbose
                self.define_aperture()
                self.initialize_aperture()
                self.move_aperture_absolute(self.cfg.lab_best_aperture_position)
                self._aperture_pos, self._aperture_f_val = self.get_aperture_position()
                status = True
                break
            except Exception as e:
                self.log.error(f'Initialising focuser: Error retry {retry}: {e}')
            self.ser = None
            retry += 1

        if status:
            logit('Focuser Initialized Successfully.', color='green')
        else:
            logit('Focuser NOT Initialized.', color='red')
            self.log.error('Focuser NOT Initialized.')
            self.log.warning('Error with serial port. Moving on.')

    def power_cycle(self, power_cycle_wait: float):
        """
        Power cycle for the focuser.

        This function pauses execution for a specified duration to simulate a power cycle,
        then reinitialize the focuser.

        Args:
            power_cycle_wait (float): The number of seconds to wait before reinitializing the camera.
        """
        self.log.debug('Power Cycle Function')

        time.sleep(power_cycle_wait)

        self.log.debug('Reinitializing Focuser')
        self.__init__(self.cfg)
        logit(f'Focuser reinitialized.', color='magenta')

    def is_open(self):
        return self.ser is not None and self.ser and self.ser.is_open

    def response_mode(self, verbose_mode: int = 1, protocol_mode: int = 0):
        """
        5.34 Set Response Modes (rm)
        Command Type : Legacy/New
        Syntax : rm <verbose{0, 1}>[, <new{0, 1}>]
        Returns : OK or 0
        Description
        This command is used to set the response modes. The first argument can be 0 or
        1, representing terse mode or verbose mode, respectively. In verbose mode all characters
        sent to the device are echoed back out and most commands return confirmations and full
        status strings. In terse mode, no characters are echoed and command responses are
        limited to error codes for legacy behavior. The second argument is optional (if not
        supplied it defaults to 0), and indicates the protocol mode – legacy (0) or new (1). Legacy
        mode maintains protocol compatibility with libraries version 15 and earlier.
        The response modes are set on a per-port basis, the change being effective for the
        port over which the command is sent.
        Examples:
        rm0,1 <CR>
        OK
        rm0,0 <CR>
        0
        """
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return "0"

        # Set response mode: rm<verbose_mode>,<protocol_mode>
        cmd = f'rm{verbose_mode},{protocol_mode}\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()  # Expected responses: b'rm0,1\rOK\r' or b'rm0,0\r0\r'
        parts = out.decode('ascii').split('\r')
        logit(f"Focuser RM (Response Mode): {parts}", color='magenta')

        # Extract the response (either "OK" or "0")
        response = parts[1] if len(parts) > 1 else parts[0]
        logit(f'  Response mode - verbose: {verbose_mode}, protocol: {protocol_mode}, result: {response}',
              color='yellow')

        return response

    def generate_f_stops(self, f_min: str, num_stops: int, f_max: str):
        """
        Generates a list of f-stops based on the minimum f-stop,
        the number of stops, and the assumption that each stop is a
        1/4 step between the minimum and maximum.

        Args:
            f_min (str): The minimum f-stop value as a string (e.g., "f14").
            num_stops (int): The total number of stops in the series.

        Returns:
            list: A list of f-stop values as strings (e.g., ["f14.0", "f14.25", ..., "f160.0"]).
                  Returns an empty list if the input f_min_str is not in the expected format.
        """
        try:
            f_min = float(f_min.replace("f", ""))
            f_max = float(f_max.replace("f", ""))
        except ValueError:
            print(f"Error: Invalid f_min_str format: '{f_min}'. Expected 'f' followed by a number.")
            return []

        f_stops = []
        step_size = (f_max - f_min) / (num_stops - 1) if num_stops > 1 else 0

        for i in range(num_stops):
            f_value = f_min + i * step_size
            f_value_str = f"f{f_value:.0f}" if f_value*100 == int(f_value)*100 else f"f{f_value:.2f}"
            f_stops.append(f_value_str)  # Format to two decimal places for clarity

        return f_stops

    def define_aperture(self):
        """
        5.2 Define Aperture (da)
        Command Type : Legacy
        Syntax : da
        Returns : f<min>,<num_stops>,f<max>
        Description
        This command will print the aperture range of the connected lens. The range
        is printed in three parts as follows:
        fmin,num_stops,fmax
        The item printed is the letter 'f' followed by the current minimum f-number. This is
        followed by a comma and the total number of stops in ¼ stops that can be commanded to
        the lens. This is then followed by another comma, the letter 'f', and the current maximum
        f-number.
        Example:
        da <CR>
        f28,28,f320
        """
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0, 0, 0
            # define aperture: -> fmin,num_stops,fmax
        cmd = 'da' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()  # b'da\rOK\rf14,28,f160\r'
        parts = out.decode('ascii').split('\r')
        logit(f"Focuser DA (Define Aperture): {parts}", color='magenta')
        self.f_min, self.num_stops, self.f_max, *_ = parts[2].split(',')
        # parts = out.decode('ascii').split('\r')
        # curr_focus_pos = int(parts[-2])
        self.num_stops = int(self.num_stops)
        logit(f'  f_min: {self.f_min} num_stops: {self.num_stops} f_max: {self.f_max}', color='yellow')
        self.f_stops = self.f_stop_sequence # self.generate_f_stops(self.f_min, self.num_stops, self.f_max)
        logit(f'  f stops: {self.f_stops}', color='yellow')

    def initialize_aperture(self):
        """
        5.16 Initialize Aperture (in)
        Command Type : Legacy
        Syntax : in
        Returns : DONE
        Description
        This command initializes the aperture motor and position. This opens up the
        aperture to its minimum f-number (maximum opening). This position is then set as the
        zero position for all subsequent aperture movement commands. This is required before
        any of the aperture movement commands can be used, as driving the aperture motor too
        far without knowing its position could damage the aperture mechanism. The library
        prevents you from issuing aperture movement commands unless the position has first
        been initialized with this command. The library can be configured to automatically
        initialize the aperture motor when a lens is attached.
        Example:
        in <CR>
        DONE
        """
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0
        cmd = 'in' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()  # b'in\rOK\rDONE\r'
        parts = out.decode('ascii').split('\r')
        logit(f"Focuser IN (Initialize Aperture): {parts}", color='magenta')
        self._aperture_pos = 0
        self._aperture_f_val = self.f_stops[0]

    def move_aperture_absolute(self, pos):
        """
        5.24 Move Aperture Absolute (ma)
        Command Type : Legacy
        Syntax : ma <pos>
        Returns : DONE<rpos>,f<f_number>
        Description
        This command moves the aperture mechanism to the specified encoder position,
        specified in ¼ stops from the fully-open position, indicated by the user in <pos>. If an
        input value would move the aperture out of the legal range the value is set to the
        boundary (i.e. min/max aperture). <rpos> is the actual position the aperture moved to in
        ¼ stops from the fully-open position, and <f_number> is the absolute position given as
        the lens f-number times ten.
        Examples:
        ma0 <CR>
        DONE0,f35
        ma5 <CR>
        DONE5,f54
        """
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0
            # define aperture: -> fmin,num_stops,fmax
        cmd = f'ma {pos}\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()  # b'ma\rOK\r8330\r'
        parts = out.decode('ascii').split('\r')
        logit(f"Focuser MA (Move Aperture Absolute): {parts}", color='magenta')
        # Focuser MA (Move Aperture Absolute): ['ma 0', 'OK', 'DONE0,f14', '']
        pos, self._aperture_f_val = parts[-2].replace('DONE', '').split(',')
        self._aperture_pos = int(pos)
        return self._aperture_pos, self._aperture_f_val

    def get_aperture_position(self):
        """
        Command Type : Legacy
        Syntax : pa
        Returns : <pos>,f<f_number>
        Description
        This command prints the current position of the aperture. <pos> is the absolute
        position given in ¼ stops from the fully-open position and <f_number> is the absolute
        position given as the lens f-number times ten.
        Example:
        pa <CR>
        5,f54
        """
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0, '??'
            # define aperture: -> fmin,num_stops,fmax
        try:
            cmd = f'pa\r'
            self.ser.write(cmd.encode('ascii'))
            out = self.ser.readline()  # b'pa\rOK\r5,f54\r'
            parts = out.decode('ascii').split('\r')
            pos, self._aperture_f_val = parts[-2].replace('DONE', '').split(',')
            logit(f"Focuser PA (Print Aperture Position): {parts} pos: {pos} f_val: {self._aperture_f_val}", color='magenta')

            self._aperture_pos = int(pos)
        except Exception as e:
            return 0, '??'
        return self._aperture_pos, self._aperture_f_val

    @property
    def aperture_pos(self):
        if self._aperture_pos is None:
            self.get_aperture_position()
        return self._aperture_pos

    @property
    def aperture_f_val(self):
        if self._aperture_f_val is None:
            self.get_aperture_position()
        return self._aperture_f_val


    @property
    def focus_position(self):
        if self._focus_position is None:
            self.get_focus_position()
        return self._focus_position

    def get_focus_position(self):
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0
        # get current focus position.
        cmd = 'pf' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()  # b'pf\rOK\r8330\r'
        parts = out.decode('ascii').split('\r')
        curr_focus_pos = int(parts[-2])
        self._focus_position = curr_focus_pos
        return curr_focus_pos

    def move_focus_position(self, new_position):
        self.server.write(f'Moving focus position to: {new_position}.')
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0
        # move focus to desired position.
        cmd = 'fa' + str(int(round(new_position))) + '\r'
        self.ser.write(cmd.encode('ascii'))
        output = self.ser.readline()
        if len(output.decode().split("E")) == 1:
            self.log.error('Focuser error, got invalid or no response for move_focus_position.')
            return 0
        curr_focus_pos = int(output.decode().split("E")[1].split(",")[0])
        # self.log.info(f'In move_focus_position function: Focus moved to: {curr_focus_pos}')
        logit(f'Focus moved to position: {curr_focus_pos}', color='magenta')
        self._focus_position = curr_focus_pos
        return curr_focus_pos

    def open_aperture(self):
        t0 = time.monotonic()
        self.server.write('Opening aperture.')
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return
        cmd = 'in' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()
        self.log.info(out.decode('ascii'))
        self.aperture_position = 'opened'
        logit(f'Aperture opened in {get_dt(t0)}.', color='magenta')

    def home_lens_focus(self):
        t0 = time.monotonic()
        self.server.write('Homing lens.')
        if not self.is_open():
            self.server.write('Focuser connection closed.', 'warning')
            self.log.warning('Focuser connection closed.')
            return 0, 0
        # self.server.write('Getting current focus position.')
        # starting_focus_position = self.get_focus_position()
        # self.server.write(f'Starting focus position: {starting_focus_position}')
        self.server.write('Moving to zero stop of lens.')
        self.server.write('Current focus position was...')
        output = self.move_focus_to_zero()
        if len(output.decode().split("E")) == 1:
            self.log.error('Focuser error, got invalid or no response for move_focus_to_zero.')
            return 0, 0
        min_focus_position = int(output.decode().split("E")[1].split(",")[0])
        # self. server.write(f'Min Focus Position: {min_focus_position}')
        starting_focus_position = abs(min_focus_position)
        self.server.write(f'Starting focus position: {starting_focus_position}')
        self.server.write(f'Old Min Focus Position: {min_focus_position}')
        self.server.write('Set zero position as zero counts.')
        output = self.set_focus_to_zero()
        min_focus_position = self.get_focus_position()
        self.server.write(f'New Min Focus Position: {min_focus_position}')
        # self.server.write('Move focus to infinity position and store encoder counts')
        self.server.write('Move focus to infinity position.')
        output = self.move_focus_to_infinity()
        if len(output.decode().split("E")) == 1:
            self.log.error('Focuser error, got invalid or no response for move_focus_to_infinity.')
            return 0, 0
        max_focus_position = int(output.decode().split("E")[1].split(",")[0])
        self.server.write(f'New Max Focus Position: {max_focus_position}')
        self.server.write('Move back to initial focus position. Now with offsets applied.')
        # output = self.move_focus_position(starting_focus_position - min_focus_position)
        output = self.move_focus_position(starting_focus_position)
        focus_position = self.get_focus_position()
        # self.server.write(f'Moved focus to: {output}')
        self.server.write(f'Moved focus to: {focus_position} in {get_dt(t0)}.')
        return min_focus_position, max_focus_position

    def close_aperture(self):
        t0 = time.monotonic()
        self.server.write('Closing aperture.')
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return
        cmd = 'in' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()
        self.server.write(out.decode('ascii'))
        self.aperture_position = 'closed'
        logit(f'Aperture closed in {get_dt(t0)}.', color='magenta')

    def move_focus_to_zero(self):
        """
        Move Focus to Zero (mz)
        Command Type : Legacy
        Syntax : mz
        Returns : DONE<signed_num_counts>,<flag>
        Description
            This command moves the lens focus to the zero position. The actual number of
            counts moved as reported by the lens' encoder is given in <signed_num_counts>. <flag>
            is 1 if the lens reports having hit a stop, 0 if it hasn't. Note that some lenses do not return
            a 1 until the second time the stop is hit.
        Example:
            mz <CR>
            DONE-1246,1
        """
        self.server.write('Move focus to zero.')
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0
        # read & store current focus position and limits
        cmd = 'mz' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()
        parts = out.decode('ascii').split('\r')
        logit(f'Moved focus to 0. {parts}', color='magenta')
        self._focus_position = 0
        return out

    def move_focus_to_infinity(self):
        """
        Move Focus to Infinity (mi)
        Command Type : Legacy
        Syntax : mi
        Returns : DONE<signed_num_counts>,<flag>
        Description
            This command moves the lens focus to the infinity position. The actual number of
            counts moved as reported by the lens' encoder is given in <signed_num_counts>. <flag>
            is 1 if the lens reports having hit a stop, 0 if it hasn't. Note that some lenses do not return
            a 1 until the second time the stop is hit.

        Example:
            mi <CR>
            DONE246,1
        """
        self.server.write('Move focus to infinity.')
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0
        # read & store current focus position and limits
        cmd = 'mi' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()
        parts = out.decode('ascii').split('\r')
        logit(f'Moved focus to inf.  {parts}', color='magenta')
        self._focus_position = self.focus_inf
        return out

    def set_focus_to_zero(self):
        if not self.is_open():
            self.log.warning('Focuser connection closed.')
            return 0
        # read & store current focus position and limits
        cmd = 'sf0' + '\r'
        self.ser.write(cmd.encode('ascii'))
        out = self.ser.readline()
        logit(f'Set focus to 0.', color='magenta')
        self._focus_position = 0
        return out

    def get_signed_num_counts(self, input) -> (str, int, int):
        """
        Input format: <prefix>\rOK\rDONE<signed_num_counts>,<flag>\r
        Example inputs:
        1. b'mz\rOK\rDONE0,1\r'
        2. b'mi\rOK\rDONE8808,1\r'

        :return: (status, signed_num_counts, flag)
        :rtype: str, int, int
        """
        status = 'ERROR'
        signed_num_counts, flag = 0, 0  # Default return values in case of error

        try:
            # 1. Decode if input is bytes
            # print(f'    RAW: {input}')
            input_str = input.decode('ascii').strip() if isinstance(input, bytes) else input.strip()
            # print(f'DECODED: {input_str}')

            # 2. Split into parts using \r as separator
            parts = [p for p in input_str.split('\r') if p]  # Remove empty parts
            if len(parts) < 3:
                return 'ERROR', 0, 0

            # 3. Get status (should be 'OK') and DONE part
            status = parts[1]  # The part after prefix and before DONE
            if status != 'OK':
                return 'ERROR', 0, 0

            done_part = parts[2]
            if not done_part.startswith('DONE'):
                return 'ERROR', 0, 0

            # 4. Extract the numeric values after DONE
            data_part = done_part[4:]  # Remove 'DONE'
            num_parts = data_part.split(',')
            if len(num_parts) != 2:
                return 'ERROR', 0, 0

            # 5. Convert to integers
            signed_num_counts = int(num_parts[0])
            flag = int(num_parts[1])
            status = 'OK'

        except (AttributeError, ValueError, IndexError, UnicodeDecodeError) as e:
            print(f'Error parsing input: {e}')
            status, signed_num_counts, flag = 'ERROR', 0, 0

        return status, signed_num_counts, flag

    def check_lens_focus(self):
        """
        Execute MZ MZ MI MI and report total number of steps
        :return:
        :rtype:
        """
        commands = ['mz', 'mz', 'mi', 'mi']
        current_focus_position = self.focus_position
        results = []
        for command in commands:
            output = ''
            if command == 'mz':
                output = self.move_focus_to_zero()
            elif command == 'mi':
                output = self.move_focus_to_infinity()

            status, signed_num_counts, flag = self.get_signed_num_counts(output)
            # Report
            results.append({'command': command, 'status': status, 'signed_num_counts': signed_num_counts, 'flag': flag})

        # Move to original focus position
        self.move_focus_position(current_focus_position)
        self.log.debug(f'Check lens results: {results}')
        return results


def test():
    # !/usr/bin/env python3
    import time, serial
    PORT = "/dev/serial/by-id/usb-FTDI_Chipi-X_FT2RA78W-if00-port0"
    CMD = 'mz'
    CMD = CMD.encode('ascii')
    BAUD = 115200  # try also 9600, etc

    def try_combo(baud, dtr, rts, rtscts, dsrdtr, term=b"\r"):
        print("Trying", baud, "DTR", dtr, "RTS", rts, "rtscts", rtscts, "dsrdtr", dsrdtr, "term", term)
        try:
            s = serial.Serial(PORT, baudrate=baud, timeout=1.0,
                              rtscts=rtscts, dsrdtr=dsrdtr,
                              bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
        except Exception as e:
            print("open failed:", e);
            return
        try:
            s.setDTR(dtr);
            s.setRTS(rts);
            time.sleep(0.05)
            s.reset_input_buffer();
            s.reset_output_buffer()
            s.write(CMD + term);
            s.flush()
            data = s.read(4096)
            print("read:", data)
        finally:
            s.close()

    # try a few combos
    for baud in (115200, 57600, 38400, 19200, 9600):
        for dtr in (False, True):
            for rts in (False, True):
                try_combo(baud, dtr, rts, False, False)


# Example usage
if __name__ == "__main__":
    logit('Running Focuser STANDALONE.')

    if False:
        test()
        sys.exit(0)

    try:
        class cfg:
            focuser_port = '/dev/ttyS0'
            baud_rate = 115200
        focuser = Focuser(cfg)  # Replace with your port and baud rate
        # ... (your application logic) ...
        results = focuser.check_lens_focus()
        print(results)

    except Exception as e:
        logit(f"Error: {e}")
    finally:
        logit(f'Exiting.', color='green')
# Last line
