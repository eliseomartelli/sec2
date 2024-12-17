.PHONY: ca-certificate
ca-certificate:
	openssl genpkey -out CA.key -algorithm RSA -pkeyopt rsa_keygen_bits:2048
	openssl rsa -pubout -in CA.key -out CA.pub
	openssl req -key CA.key -new -x509 -days 365 -out ca-bundle.crt \
		-subj="/C=IT/ST=Torino/L=Torino/O=Universita  di Torino/CN=CA di Eliseo Martelli/emailAddress=ca@eliseomartelli.it"

.PHONY: server-certificate
server-certificate:
	openssl genpkey -out server.key -algorithm RSA -pkeyopt rsa_keygen_bits:2048
	openssl rsa -pubout -in server.key -out server.pub
	openssl req -key server.key -new -out server.req \
		-subj="/C=IT/ST=Torino/L=Torino/O=Universita  di Torino/CN=www.plubmers.it/emailAddress=ca@eliseomartelli.it"
	openssl x509 -days 365 -CA ca-bundle.crt -CAkey CA.key -CAcreateserial -CAserial ca.srl -req -in server.req -out server.crt

.PHONY: client-certificate
client-certificate:
	openssl genpkey -out client.key -algorithm RSA -pkeyopt rsa_keygen_bits:2048
	openssl rsa -pubout -in client.key -out client.pub
	openssl req -key client.key -new -out client.req \
		-subj="/C=IT/ST=Torino/L=Torino/O=Universita  di Torino/CN=Eliseo Martelli/"
	openssl x509 -days 365 -CA ca-bundle.crt -CAkey CA.key -CAcreateserial -CAserial ca.srl -req -in client.req -out client.crt
	openssl pkcs12 -export -out client.p12 -inkey client.key -in client.crt -certfile ca-bundle.crt





