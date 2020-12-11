const { Readability, isProbablyReaderable } = require('@mozilla/readability');
const jsdom = require('jsdom');
const { JSDOM } = jsdom;

export interface Features {
  title: string;
  byline: string;
  content: string;
  textContent: string;
  length: number;
  siteName: string;
  url: string;
}

export async function extract(url: string): Promise<Features> {
  const doc = await JSDOM.fromURL(url);
  if (!isProbablyReaderable(doc.window.document)) {
    throw new Error('Not readable');
  }
  const parsed = new Readability(doc.window.document).parse();

  parsed.url = url;
  delete parsed.excerpt;
  delete parsed.dir;
  delete parsed.byline;

  return parsed;
}
