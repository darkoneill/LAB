# ============================================================
# OpenClaw - Nginx Reverse Proxy (custom image)
# Embarque la config directement pour eviter les bind mount issues
# ============================================================
FROM nginx:alpine

# Supprimer la config par defaut
RUN rm -f /etc/nginx/conf.d/default.conf

# Copier notre config
COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/ /etc/nginx/conf.d/

# Creer le repertoire de cache
RUN mkdir -p /var/cache/nginx/static \
    && chown -R nginx:nginx /var/cache/nginx
