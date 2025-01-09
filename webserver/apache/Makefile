.PHONY: ca-certificate
ca-certificate:
	# Generate the Certificate Authority (CA) private key
	openssl genpkey -out CA.key -algorithm RSA -pkeyopt rsa_keygen_bits:2048

	# Extract the public key from the CA private key
	openssl rsa -pubout -in CA.key -out CA.pub

	# Create the self-signed certificate for the CA
	openssl req -key CA.key -new -x509 -days 365 -out ca-bundle.crt \
		-subj "/C=IT/ST=Torino/L=Torino/O=Universita di Torino/CN=CA di Eliseo Martelli/emailAddress=ca@eliseomartelli.it"

.PHONY: server-certificate
server-certificate:
	# Generate the server private key
	openssl genpkey -out server.key -algorithm RSA -pkeyopt rsa_keygen_bits:2048
	
	# Generate the CSR (Certificate Signing Request) for the server using the custom config file
	openssl req -new -key server.key -out server.req -config ./openssl.conf

	# Sign the server CSR with the CA private key to generate the server certificate
	openssl x509 -req -days 365 -in server.req -CA ca-bundle.crt -CAkey CA.key -CAcreateserial -CAserial ca.srl -out server.crt

	# Clean up the CSR file as it's no longer needed
	rm server.req

.PHONY: client-certificate
client-certificate:
	# Generate the client private key
	openssl genpkey -out client.key -algorithm RSA -pkeyopt rsa_keygen_bits:2048
	
	# Extract the public key from the client private key
	openssl rsa -pubout -in client.key -out client.pub
	
	# Generate the CSR for the client using the custom config file
	openssl req -key client.key -new -out client.req -config openssl.conf -extensions v3_req \
		-subj "/C=IT/ST=Torino/L=Torino/O=Universita di Torino/CN=Eliseo Martelli/"
	
	# Sign the client CSR with the CA private key to generate the client certificate
	openssl x509 -days 365 -CA ca-bundle.crt -CAkey CA.key -CAcreateserial -CAserial ca.srl -req -in client.req -out client.crt
	
	# Convert the client certificate and private key to PKCS#12 format
	openssl pkcs12 -export -out client.p12 -inkey client.key -in client.crt -certfile ca-bundle.crt

.PHONY: invalid-client-certificate
invalid-client-certificate:
	# Generate the client private key
	openssl genpkey -out invalid-client.key -algorithm RSA -pkeyopt rsa_keygen_bits:2048
	
	# Extract the public key from the client private key
	openssl rsa -pubout -in invalid-client.key -out invalid-client.pub
	
	# Generate the CSR for the client using the custom config file
	openssl req -key invalid-client.key -new -out invalid-client.req -config openssl.conf -extensions v3_req \
		-subj "/C=IT/ST=Torino/L=Torino/O=Universita di Torino/CN=Invalid Martelli/"
	
	# Sign the client CSR with the CA private key to generate the client certificate
	openssl x509 -days 365 -CA ca-bundle.crt -CAkey CA.key -CAcreateserial -CAserial ca.srl -req -in invalid-client.req -out invalid-client.crt
	
	# Convert the client certificate and private key to PKCS#12 format
	openssl pkcs12 -export -out invalid-client.p12 -inkey invalid-client.key -in invalid-client.crt -certfile ca-bundle.crt
