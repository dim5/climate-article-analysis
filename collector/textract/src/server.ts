import Koa from 'koa';
import json from 'koa-json';

import yamljs from 'yamljs';
import { koaSwagger } from 'koa2-swagger-ui';

import { config } from './config';
import router from './routing';

const app = new Koa();

app.use(async (ctx, next) => {
  try {
    await next();
  } catch (err) {
    ctx.status = err.status || 500;
    ctx.body = err.message;
    ctx.app.emit('error', err, ctx);
  }
});

app.use(json({ pretty: false }));
app.use(router.routes()).use(router.allowedMethods());

const spec = yamljs.load('./openapi.yaml');
app.use(koaSwagger({ routePrefix: '/swagger', swaggerOptions: { spec } }));

app.listen(config.port, () => {
  console.log(`Listening on ${config.port}`);
});
