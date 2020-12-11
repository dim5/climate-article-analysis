using System;
using System.Diagnostics.CodeAnalysis;

namespace Data.DTO
{
    public class RSSResult : IEquatable<RSSResult>
    {
        public Uri URL { get; }
        public DateTimeOffset PublishDate { get; }

        public RSSResult(Uri url, DateTimeOffset publishDate)
        {
            URL = url;
            PublishDate = publishDate;
        }

        public bool Equals([AllowNull] RSSResult other) => other?.URL == URL;

        public override int GetHashCode()
        {
            return HashCode.Combine(URL);
        }
    }
}
