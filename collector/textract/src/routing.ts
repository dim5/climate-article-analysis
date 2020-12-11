import Router from "@koa/router";
import { extract } from "./extractor";
const router = new Router();

router.get("/extract", async ctx => {
  const url = ctx.query["url"];

  const fail = (message?: string, code = 400, error?: Error) => {
    ctx.status = code;
    if (!!error) {
      console.error(error);
    }
    ctx.body = { message };
  };

  if (!url) {
    fail("No url provided", 404);
    return;
  }

  try {
    const feat = await extract(url);
    ctx.body = feat;
  } catch (e) {
    fail(e.message, 400, e);
  }
});

router.get("/healthcheck", ctx => {
  ctx.body = "OK";
});

export default router;
