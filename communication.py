import sys
import socket


class Communication(object):

    def __init__(self, private_ip, public_ip):
        self._private_ip = private_ip.split(':')[0]
        self._private_port = int(private_ip.split(':')[1])
        self._public_ip = public_ip.split(':')[0]
        self._public_port = int(public_ip.split(':')[1])

    def start_socket_ps(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self._private_ip, self._private_port))
        server_socket.listen(5)
        return server_socket

    def start_socket_client(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        client_socket.connect((self._public_ip, self._public_port))
        return client_socket

    def receiving_subroutine(self, connection_socket):
        timeout = 1.0
        while True:
            ultimate_buffer = b''
            connection_socket.settimeout(240)
            first_round = True
            while True:
                try:
                    receiving_buffer = connection_socket.recv(8192*2)
                except Exception as e:
                    if str(e) != 'timed out':
                        print(e)
                        sys.stdout.flush()
                    break
                if first_round:
                    connection_socket.settimeout(timeout)
                    first_round = False
                if not receiving_buffer:
                    break
                ultimate_buffer += receiving_buffer
            if(ultimate_buffer[0:5]==b'10111'):
                message = ultimate_buffer[5:int(0 - int(5))]
                connection_socket.send(b'RECEIVED')
                print('Recived right message!')
            else:
                connection_socket.send(b'ERRORrrr')
                print("Received wrong message!")
                continue
            return message

    def get_message(self, connection_socket):
        message = self.receiving_subroutine(connection_socket)
        return message

    def send_message(self, message_to_send, connection_socket):
        message = b'10111'+ message_to_send + b'EOF\r\n'
        connection_socket.settimeout(240)
        connection_socket.sendall(message)
        while True:
            check = connection_socket.recv(8)
            print(check)
            if check == b'ERRORrrr':
                connection_socket.sendall(message)
            elif check == b'RECEIVED':
                break
