using System;
using Data.DTO;

namespace Data.Model
{
    public class Article
    {
        public int Id { get; set; }
        public string Title { get; set; }

        public string Content { get; set; }

        public string TextContent { get; set; }

        public string SiteName { get; set; }

        public Uri Url { get; set; }

        public DateTimeOffset Created { get; set; }

        public static Article FromExtract(Extract e) => FromExtract(e, DateTimeOffset.UtcNow);

        public static Article FromExtract(Extract e, DateTimeOffset createDate) => new Article
        {
            Content = e.Content,
            SiteName = e.SiteName,
            Title = e.Title,
            Url = e.Url,
            TextContent = e.TextContent,
            Created = createDate
        };
    }
}