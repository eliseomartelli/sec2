# Eliseo martelli
# Inizio la configurazione del web server partendo dalla distribuzione
# Alpine Linux.
FROM alpine:3.20.3

# Vado quindi ad installare le dipenenze necessarie affinchè il server
# funzioni correttamente.
RUN apk add --no-cache \
		apache2 \
		apache2-utils \
		apache2-ssl \
		php83 \
		php83-apache2 \
		php83-ldap \
		;

# Cambio la porta del server e imposto la 443, cambio il nome del server
# in localost.
RUN sed -i 's/^Listen 80/Listen 443/' /etc/apache2/httpd.conf && \
	sed -i 's/^#ServerName.*/ServerName www.plumbers.it/' /etc/apache2/httpd.conf && \
	sed -i 's/AllowOverride None/AllowOverride All/' /etc/apache2/httpd.conf \
	;

EXPOSE 8080

WORKDIR /var/www/localhost/htdocs

COPY index.html index.html
COPY protectedroute/ protectedroute/

# Creo un file delle password e ci insereisco un utente.
RUN htpasswd -cbB /etc/apache2/passwords \
	testuser \
	testuser && \
	\
	htpasswd -bB /etc/apache2/passwords \
	mario \
	mario && \
	\
	htpasswd -bB /etc/apache2/passwords \
	luigi \
	luigi \
	;

# Aggiungo il gruppo plumbers.
RUN echo "plumbers: mario luigi" > /etc/apache2/groups

# Aggiungo la plumbers route, sarà accessibile solo da mario e luigi.
COPY plumbersroute/ plumbersroute/

# Creo la cartella e metterò dento certificati.
RUN mkdir -p /etc/ssl/mycerts
COPY ./server.key /etc/ssl/mycerts/server.key
COPY ./server.crt /etc/ssl/mycerts/server.crt
COPY ./ca-bundle.crt /etc/ssl/mycerts/ca-bundle.crt

RUN chown root:root /etc/ssl/mycerts/server.key /etc/ssl/mycerts/server.crt
RUN chmod 600 /etc/ssl/mycerts/server.key
RUN chmod 644 /etc/ssl/mycerts/server.crt

COPY ./ssl.conf /etc/apache2/conf.d/ssl.conf

COPY ./server.conf /etc/apache2/conf.d/server.conf

COPY certroute/ certroute/
COPY ldapclient/ ldapclient/

CMD ["httpd", "-D", "FOREGROUND"]
